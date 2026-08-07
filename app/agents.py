import json
import os
import time
from typing import Optional, AsyncGenerator

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .tools import pubmed_search, configure_pubmed

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
            tools=[self.pubmed_tool],
            description="Executes PubMed searches and reports results",
            system_message="""Execute pubmed_search for each query provided. Deduplicate results. Report the total number of unique papers found and list their titles, authors, and years. Do not paste full abstracts.""",
        )

        # 3. 综述生成
        self.synthesis_agent = AssistantAgent(
            name="SynthesisAgent",
            model_client=self.model_client,
            description="Medical literature review writer",
            system_message="""You are an academic writer. Based **ONLY** on the search results provided by SearchAgent, write a comprehensive literature review on the given topic.

        **You MUST organize your response using exactly the following 7 numbered sections, with each section title as a Markdown heading (##). Do not omit any section. Do not merge sections.**

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

        **Important:** Write in a formal, academic tone. Use Markdown for lists and emphasis. The review should be thorough and well‑organized. End with **SYNTHESIS_COMPLETE**.""",
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