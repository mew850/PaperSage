import json
import os
import time
from typing import Optional, AsyncGenerator, List, Dict, Any
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
import requests

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


# ---------- 语义检索 Agent（深度模式，支持交互式追问） ----------
class SemanticSearchAgent:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.exa_api_key = os.getenv("EXA_API_KEY")
        if not self.exa_api_key:
            print("⚠️  EXA_API_KEY 未设置，Exa搜索不可用")
        self.exa = Exa(self.exa_api_key) if self.exa_api_key else None

        # 对话历史存储（用于交互式问答）
        self.conversation_history = []
        # 存储最近一次检索的论文列表
        self._last_papers = []

        # LLM 配置（用于追问回答）
        self.llm_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.llm_base_url = base_url or os.getenv("LITERAS_BASE_URL") or "https://api.deepseek.com/v1"
        self.llm_model = model or os.getenv("LITERAS_MODEL") or "deepseek-chat"
        self.client = AsyncOpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url) if self.llm_api_key else None

    def get_last_papers(self) -> List[dict]:
        """获取最近一次检索的论文列表"""
        return self._last_papers

    # ---------- Exa Agent 智能检索 ----------
    async def _exa_agent_search(self, query: str) -> dict:
        """将用户原始 query 直接传给 Exa Agent，要求以 JSON 格式返回文献列表。"""
        if not self.exa:
            return None

        prompt = f"""用户要求：{query}

请根据上述要求检索相关学术文献，并以 JSON 格式返回结果。
JSON 格式如下（每篇文献尽可能包含 title, authors, year, journal, impact_factor, doi, abstract）：
{{"papers": [{{"title": "", "authors": "", "year": "", "journal": "", "impact_factor": "", "doi": "", "abstract": ""}}]}}
"""
        try:
            loop = asyncio.get_running_loop()
            run = await loop.run_in_executor(
                None,
                lambda: self.exa.agent.runs.create(query=prompt)
            )
            completed_run = await loop.run_in_executor(
                None,
                lambda: self.exa.agent.runs.poll_until_finished(run.id)
            )
            if completed_run.output and completed_run.output.text:
                text = completed_run.output.text
                import json, re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    data = json.loads(text)
                return data
            return None
        except Exception as e:
            print(f"Exa Agent 检索异常: {e}")
            return None

    # ---------- 格式化结果 ----------
    def _format_agent_result(self, data: dict, query: str) -> str:
        """将结构化数据转换为 Markdown 表格"""
        papers = data.get('papers', [])
        if not papers:
            return f"## 🔍 检索结果：{query}\n\n未找到相关结果。"

        headers = ['标题', '作者', '年份', '期刊', '影响因子', 'DOI/链接']
        keys = ['title', 'authors', 'year', 'journal', 'impact_factor', 'doi']

        table = "| " + " | ".join(headers) + " |\n"
        table += "|" + "|".join(["------" for _ in headers]) + "|\n"

        for p in papers[:10]:
            row = []
            for key in keys:
                val = p.get(key, '')
                if val == '' or val is None:
                    val = '-'
                else:
                    val = str(val).replace('\n', ' ').replace('|', '\\|')
                row.append(val)
            table += "| " + " | ".join(row) + " |\n"

        abstract_section = "\n**摘要详情**：\n"
        for idx, p in enumerate(papers[:10], 1):
            title = p.get('title', '无标题')
            abstract = p.get('abstract', '无摘要')
            if len(abstract) > 300:
                abstract = abstract[:300] + '…'
            abstract_section += f"{idx}. **{title}**\n   {abstract}\n\n"

        return f"## 🔍 检索结果：{query}\n\n{table}\n{abstract_section}"

    # ---------- 主搜索接口 ----------
    async def search(self, query: str, use_history: bool = False) -> str:
        """
        执行深度检索（Agent），保存结果到 _last_papers，返回格式化的表格。
        """
        data = await self._exa_agent_search(query)
        if data and data.get('papers'):
            papers = data['papers']
            self._last_papers = papers
            result = self._format_agent_result(data, query)
            if use_history:
                self.conversation_history.append({"role": "user", "content": query})
                self.conversation_history.append({"role": "assistant", "content": result})
            return result
        else:
            return f"## 🔍 检索结果：{query}\n\n❌ 检索失败，请稍后重试。"

    # ---------- 基于检索结果的交互式问答 ----------
    async def ask_with_results(self, question: str, papers: List[dict]) -> str:
        """
        基于给定的论文列表回答用户问题（使用 LLM）。
        """
        if not papers:
            return "没有可用的检索结果，请先进行语义检索。"

        # 构建上下文：取前5篇论文的标题和摘要片段
        context = "以下是最近检索到的学术文献信息：\n\n"
        for i, p in enumerate(papers[:5], 1):
            title = p.get('title', '无标题')
            abstract = p.get('abstract', '无摘要')
            context += f"{i}. 标题：{title}\n"
            context += f"   摘要：{abstract[:300]}...\n\n"

        prompt = f"""你是一个科研助手，用户基于上述检索结果提出了一个问题。请根据文献内容回答用户的问题。

{context}

用户问题：{question}

请用中文回答，简洁准确，并注明引用的文献（用编号或标题）。
"""
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ 回答生成失败: {e}"

    # ---------- 交互式问答（带历史） ----------
    async def ask(self, query: str) -> str:
        """基于历史对话的交互式问答（保留原有功能）"""
        if self.conversation_history:
            context = "\n".join([
                f"{item['role']}: {item['content'][:200]}"
                for item in self.conversation_history[-6:]
            ])
            enhanced_query = f"基于以下对话历史：\n{context}\n\n当前问题：{query}"
        else:
            enhanced_query = query
        result = await self.search(enhanced_query, use_history=True)
        return result

    def clear_history(self):
        self.conversation_history = []
        self._last_papers = []