import json
import os
import time
from typing import Optional, AsyncGenerator
from openai import AsyncOpenAI
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from exa_py import Exa
from .tools import pubmed_search, configure_pubmed, exa_search
import re 
DEEPSEEK_BASE_URL = "https://llmapi.paratera.com/"
DEFAULT_MODEL = "DeepSeek-V4-Flash"

# 配置（无迭代，直接搜索 + 综述）
LEAN = {
    "max_refine": 0,
    "max_messages": 16,
    "max_words": 0,
}
FULL = {
    "max_refine": 0,
    "max_messages": 0,
    "max_words": 0,
}

def resolve_api_key(api_key: Optional[str] = None) -> Optional[str]:
    return api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")

def _word_cap(n: int) -> str:
    return ""

class AcademicSearchTeam:
    """文献检索与综述生成 Agent（无迭代校验）"""
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str = None,
        full: bool = False,
        base_url: str = None,
    ):
        self.full = full or os.getenv("LITERAS_FULL", "").lower() in ("1", "true", "yes")
        cfg = FULL if self.full else LEAN
        self.max_refine = cfg["max_refine"]
        self.max_messages = cfg["max_messages"]
        configure_pubmed(full=self.full)

        key = resolve_api_key(api_key)
        url = base_url or os.getenv("LITERAS_BASE_URL") or DEEPSEEK_BASE_URL
        model_name = os.getenv("LITERAS_MODEL") or model

        client_kwargs = {
            "model": model_name,
            "temperature": 0.7,
            "api_key": key,
            "base_url": url,
            "model_capabilities": {
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
            },
        }
        self.model_client = OpenAIChatCompletionClient(**client_kwargs)

        self.pubmed_tool = FunctionTool(
            pubmed_search,
            description="Search PubMed for academic articles and return structured results",
        )

        self.exa_tool = FunctionTool(
            exa_search,
            description="Search academic literature across all disciplines using Exa. Returns title, URL, and summary highlights. Use when the query is not biomedical or when PubMed lacks sufficient results."
        )

        # 1. 查询规划
        self.query_planner = AssistantAgent(
            name="QueryPlanner",
            model_client=self.model_client,
            description="Expert at generating comprehensive search queries",
            system_message="""You are a search query optimizer. Given a user topic, generate 2-4 concise PubMed search queries (English) that will retrieve the most relevant academic papers. Return only a JSON object: {"main_queries": ["q1", "q2", ...]}. Do not include any other text.""",
        )

        # 2. 搜索执行
        self.search_agent = AssistantAgent(
            name="SearchAgent",
            model_client=self.model_client,
            tools=[self.pubmed_tool, self.exa_tool],
            description="Executes literature searches using both PubMed and Exa, deduplicates and combines results.",
            system_message="""You are a literature search specialist with two tools:
            - pubmed_search: for biomedical/medical literature (excellent for life sciences).
            - exa_search: for general academic literature across all fields (physics, chemistry, engineering, social sciences, etc.).

            Use the appropriate tool based on the user's topic. If the query is clearly biomedical, prefer PubMed. For other topics, use Exa. You may also use both and merge the results.

            After searching, report the total number of unique papers found and list their titles, authors, and years. Do not paste full abstracts. Keep responses concise.""",
        )

        # 3. 综述生成
        self.synthesis_agent = AssistantAgent(
            name="SynthesisAgent",
            model_client=self.model_client,
            description="Literature review writer for any discipline",
            system_message="""You are an academic writer. Based **ONLY** on the search results provided by SearchAgent, write a comprehensive literature review on the given topic.

            You MUST structure your review using the following 7 numbered sections, each as a Markdown heading (##). Do not omit any section.

            ## 1. Background & Significance
            Describe the broader context, history, and importance of the research topic. Explain why it matters.

            ## 2. Literature Search Strategy
            Briefly mention how the studies were selected (e.g., databases, keywords, inclusion criteria). You may infer this from the search process.

            ## 3. Summary of Key Studies
            For each key paper, summarize:
            - Research design (e.g., experimental, observational, review)
            - Sample/population (if applicable)
            - Main findings and conclusions
            - Strengths and limitations (if mentioned)
            Organize these studies thematically or chronologically.

            ## 4. Synthesis of Findings
            Identify common themes, trends, discrepancies, or controversies across studies. Compare and contrast the evidence.

            ## 5. Gaps and Limitations
            Point out what is still unknown, contradictory, or methodologically weak in the current literature.

            ## 6. Research Objectives
            Clearly state the objective of the present review based on the identified gaps.

            ## 7. References
            Cite all references using author‑year format (e.g., AuthorYear). Include a complete reference list.

            Write in a formal, academic tone. Use Markdown for lists and emphasis. The review should be thorough and well‑organized. End with **SYNTHESIS_COMPLETE**.""",
        )

        # 4. 格式化输出
        self.formatter_agent = AssistantAgent(
            name="FormatterAgent",
            model_client=self.model_client,
            description="Formats the final output",
            system_message="""You receive the synthesis from SynthesisAgent, which already follows the required 7‑section structure. Do not reorganize or rewrite it. Only add a main title at the top (e.g., "# Literature Review: [Topic]") and ensure proper Markdown formatting. Then append **TERMINATE** at the end. Output only the final Markdown document.""",
        )

        self.current_step = 0

        def selector_func(messages):
            if not messages:
                self.current_step = 0
                return "QueryPlanner"
            last_msg = messages[-1]
            source = last_msg.source
            if source == "QueryPlanner":
                self.current_step = 1
                return "SearchAgent"
            elif source == "SearchAgent":
                self.current_step = 2
                return "SynthesisAgent"
            elif source == "SynthesisAgent":
                self.current_step = 3
                return "FormatterAgent"
            elif source == "FormatterAgent" and "TERMINATE" in str(last_msg.content).upper():
                return None
            else:
                return "QueryPlanner"

        term = TextMentionTermination("TERMINATE")
        if self.max_messages > 0:
            term = term | MaxMessageTermination(self.max_messages)

        self.team = SelectorGroupChat(
            participants=[
                self.query_planner,
                self.search_agent,
                self.synthesis_agent,
                self.formatter_agent,
            ],
            model_client=self.model_client,
            termination_condition=term,
            selector_func=selector_func,
        )

    async def process_query(self, query: str):
        """执行搜索与综述生成，返回流式更新"""
        try:
            initial_message = TextMessage(content=query, source="user")
            async for message in self.team.run_stream(task=initial_message):
                if hasattr(message, "source") and hasattr(message, "content"):
                    yield {
                        "type": "update",
                        "agent": message.source,
                        "content": message.content,
                    }
            await self.team.reset()
        except Exception as e:
            yield {"type": "error", "message": str(e)}


# ---------- 简单对话 Agent（直接调用 LLM） ----------
class SimpleChatAgent:
    """普通问答 Agent，不进行搜索"""
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        from openai import AsyncOpenAI
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LITERAS_BASE_URL") or DEEPSEEK_BASE_URL
        self.model = model or os.getenv("LITERAS_MODEL") or DEFAULT_MODEL
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def chat_stream(self, messages: list) -> AsyncGenerator[str, None]:
        """流式对话"""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.7,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class SemanticSearchAgent:
    """专门执行语义检索，返回结构化表格，摘要由 LLM 生成"""
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.exa_api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.exa_api_key:
            print("⚠️  EXA_API_KEY 未设置，语义检索不可用")
        # LLM 配置（用于生成摘要）
        self.llm_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.llm_base_url = base_url or os.getenv("LITERAS_BASE_URL") or "https://api.deepseek.com/v1"
        self.llm_model = model or os.getenv("LITERAS_MODEL") or "deepseek-chat"
        self.client = AsyncOpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url) if self.llm_api_key else None

    async def _generate_summary(self, title: str, text: str, max_len: int = 300) -> str:
        """调用 LLM 生成简洁摘要"""
        if not self.client:
            return text[:200] + "…" if text else "No summary"
        try:
            prompt = f"请用中文为以下文献标题和内容生成一个简洁的摘要（不超过{max_len}字）：\n标题：{title}\n内容片段：{text[:1000]}\n摘要："
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            summary = response.choices[0].message.content.strip()
            if len(summary) > max_len:
                summary = summary[:max_len] + "…"
            return summary
        except Exception as e:
            print(f"摘要生成失败: {e}")
            return text[:200] + "…" if text else "No summary"

    async def search(self, query: str, num_results: int = 10) -> str:
        if not self.exa_api_key:
            return "❌ 语义检索未配置，请设置 EXA_API_KEY"

        try:
            loop = asyncio.get_running_loop()
            exa = Exa(self.exa_api_key)
            result = await loop.run_in_executor(
                None,
                lambda: exa.search(
                    query,
                    num_results=num_results,
                    type="auto",
                    contents={
                        "text": {"max_characters": 1000},
                        "highlights": True
                    }
                )
            )

            if not result.results:
                return "未找到相关结果，请尝试其他关键词。"

            # 并发生成摘要
            tasks = []
            for item in result.results:
                title = item.title or "N/A"
                text = ""
                if item.highlights and len(item.highlights) > 0:
                    text = item.highlights[0]
                elif hasattr(item, 'text') and item.text:
                    text = item.text
                tasks.append(self._generate_summary(title, text))
            summaries = await asyncio.gather(*tasks)

            # 构建表格
            table = "| 标题 | 作者 | 日期 | 摘要 | URL |\n"
            table += "|------|------|------|------|-----|\n"
            for idx, item in enumerate(result.results):
                # 清洗标题
                title_raw = item.title or "N/A"
                title_cleaned = title_raw.replace('\n', ' ').replace('\r', ' ').strip()
                title_cleaned = re.sub(r'\s+', ' ', title_cleaned)
                title = title_cleaned.replace("|", "\\|")

                # 清洗作者
                author = "Unknown"
                if hasattr(item, 'author') and item.author:
                    author_raw = str(item.author)
                    author_cleaned = author_raw.replace('\n', ' ').replace('\r', ' ').strip()
                    author_cleaned = re.sub(r'\s+', ' ', author_cleaned)
                    author = author_cleaned.replace("|", "\\|")
                elif hasattr(item, 'authors') and item.authors:
                    author_raw = ", ".join(item.authors)[:50]
                    author_cleaned = author_raw.replace('\n', ' ').replace('\r', ' ').strip()
                    author_cleaned = re.sub(r'\s+', ' ', author_cleaned)
                    author = author_cleaned.replace("|", "\\|")

                # 日期
                date = "Unknown"
                if hasattr(item, 'published_date') and item.published_date:
                    date = str(item.published_date)[:10]
                elif hasattr(item, 'date') and item.date:
                    date = str(item.date)[:10]

                # 清洗摘要，若为空则赋默认值
                summary_raw = summaries[idx] if idx < len(summaries) and summaries[idx] else "无摘要"
                summary_cleaned = summary_raw.replace('\n', ' ').replace('\r', ' ').strip()
                summary_cleaned = re.sub(r'\s+', ' ', summary_cleaned)
                summary = summary_cleaned.replace("|", "\\|")

                # URL 清洗
                url = item.url or ""
                url_cleaned = url.replace("|", "\\|")

                table += f"| {title} | {author} | {date} | {summary} | [{url_cleaned}]({url_cleaned}) |\n"

            return f"## 🔍 检索结果：{query}\n\n{table}"

        except Exception as e:
            return f"❌ 检索出错：{str(e)}"