import json
import re
import asyncio
import tempfile
import shutil
import os
import io
import requests
from bs4 import BeautifulSoup
import PyPDF2
import pdfplumber
from fastapi import WebSocket
from .agents import AcademicSearchTeam, SimpleChatAgent, SemanticSearchAgent
from typing import AsyncGenerator

# ---------- 全局缓存（存储上传文件内容） ----------
file_cache = {}

# ---------- 文件解析函数 ----------
def parse_pdf(file_path: str) -> str:
    """提取 PDF 文本"""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''.join(page.extract_text() or '' for page in pdf.pages)
        if text.strip():
            return text
    except:
        pass
    # 回退到 PyPDF2
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''.join(page.extract_text() or '' for page in reader.pages)
    return text

def parse_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def parse_url(url: str) -> str:
    """从 URL 获取文本内容（支持 HTML 和 PDF）"""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '').lower()
        if 'pdf' in content_type:
            with io.BytesIO(resp.content) as pdf_io:
                with pdfplumber.open(pdf_io) as pdf:
                    text = ''.join(page.extract_text() or '' for page in pdf.pages)
            return text
        else:
            soup = BeautifulSoup(resp.text, 'html.parser')
            for s in soup(['script', 'style']):
                s.decompose()
            return soup.get_text(separator='\n')
    except Exception as e:
        return f"获取 URL 内容失败: {e}"

class ChatHandler:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.search_agent = AcademicSearchTeam(api_key=api_key, base_url=base_url, model=model)
        self.chat_agent = SimpleChatAgent(api_key=api_key, base_url=base_url, model=model)
        self.quick_search_agent = SemanticSearchAgent(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        
        # 存储每个会话的文献上下文（上传的全文）
        self.session_docs = {}
        # 存储每个会话的检索结果（用于交互式追问）
        self.session_search_results = {}

        # ---------- 意图识别关键词 ----------
        self.strong_keywords = [
            "综述", "文献", "研究进展", "最新研究", "机制", "调控", "通路",
            "review", "literature", "mechanism", "pathway", "regulate",
            "meta-analysis", "systematic review", "论文", "文章",
            "survey", "overview", "state-of-the-art", "research", "study",
            "物理", "化学", "材料", "工程", "计算机", "数学", "生物", "医学",
            "physics", "chemistry", "materials", "engineering", "computer",
            "mathematics", "biology", "medicine", "neural", "quantum"
        ]
        self.weak_keywords = [
            "影响", "作用", "发现", "报道", "实验", "研究", "基因", "蛋白",
            "effect", "role", "function", "experiment", "result",
            "analysis", "model", "algorithm", "system", "design"
        ]
        self.negation_patterns = [
            "不要", "不需要", "不用", "别", "不想", "没", "不要综述", "不查文献",
            "no review", "not search", "don't search"
        ]
        self.chat_indicators = [
            "你好", "hi", "hello", "谢谢", "thanks", "再见", "bye", "哈哈", "嗯嗯",
            "what is", "who is", "how to", "tell me about yourself"
        ]
        self.greeting_patterns = [
            "你好", "hi", "hello", "hey", "你是谁", "你叫什么", "介绍一下你自己",
            "介绍一下", "你是谁呀", "你好呀", "what's your name", "who are you"
        ]
        # 快速检索关键词（最高优先级）
        self.quick_search_keywords = [
            "检索", "搜索", "查一下", "查找", "search", "find", "look up",
            "最新新闻", "latest news", "trending"
        ]
        # 功能询问关键词
        self.function_keywords = [
            "功能", "能做什么", "有什么功能", "可以做什么", "能力", "capabilities",
            "what can you do", "functions"
        ]

    # ---------- 提取文献卡片（结构化JSON） ----------
    async def _extract_card(self, text: str, user_query: str = "") -> str:
        """
        调用LLM从文献全文提取核心信息，返回结构化卡片（Markdown格式）。
        """
        # 截断过长的文本（保留前8000字符）
        if len(text) > 8000:
            text = text[:8000] + "...(截断)"

        prompt = f"""你是一位专业的科研文献分析助手。请仔细阅读以下文献内容，提取四个核心信息，并以严格的JSON格式返回。

    要求：
    - 只返回一个JSON对象，键为 problem, method, results, limitations。
    - 每个字段的值为一段详细、专业的中文描述（**每段不超过200字**），要求内容具体，包含关键细节（如具体技术名称、实验参数、主要数据、统计结果、样本量等），避免笼统概括。
    - 不要包含任何额外文字、注释或Markdown标记。

    JSON结构示例：
    {{
    "problem": "研究旨在解决...，该问题的重要性在于...",
    "method": "采用了...技术/方法，具体步骤包括...，参数设置为...，样本量为...",
    "results": "主要发现是...，具体数据为...（如均值±标准差，p值...），统计显著性...",
    "limitations": "局限性包括...，例如样本量不足、方法适用范围限制、未考虑...等"
    }}

    文献内容：
    {text}
    """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.chat_agent.client.chat.completions.create(
                model=self.chat_agent.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            card = json.loads(content)
            for key in ["problem", "method", "results", "limitations"]:
                if key not in card:
                    card[key] = "信息缺失"
            md = f"""## 📄 文献卡片

- **核心问题**：{card['problem']}
- **研究方法**：{card['method']}
- **关键结果**：{card['results']}
- **局限性**：{card['limitations']}



---
💡 您可以继续提问关于这篇文献的任何细节，我会基于全文为您解答。
    """
            return md
        except json.JSONDecodeError:
            return "❌ 无法生成结构化卡片，请尝试重新上传或直接提问。"
        except Exception as e:
            return f"❌ 卡片生成失败：{str(e)}"

    # ---------- 文献问答（始终使用完整全文） ----------
    async def _literature_chat(self, session_id: int, user_msg: str) -> AsyncGenerator[str, None]:
        """
        基于当前会话的文献上下文进行问答，返回流式响应。
        始终使用完整全文，不做任何截断。
        """
        context = self.session_docs.get(session_id)
        if not context:
            yield "没有正在阅读的文献，请先上传文件或解析URL。"
            return

        full_text = context["text"]

        # 构建系统提示
        system_prompt = f"""你是一位科研助手，专门帮助用户理解一篇文献。请根据以下文献内容回答用户的问题。如果用户的问题涉及文献之外的内容，请委婉引导回文献本身。

文献内容（完整全文）：
{full_text}

注意：回答要准确、简洁，并引用文献中的具体信息。如果无法从文献中找到答案，请如实说明。
"""
        history = context.get("history", [])
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_msg}]

        async for chunk in self.chat_agent.chat_stream(messages):
            yield chunk

        history.append({"role": "user", "content": user_msg})

    # ---------- 已有方法：意图判断、问候等 ----------
    def _is_search_intent(self, text: str):
        lower = text.lower().strip()
        if any(neg in lower for neg in self.negation_patterns):
            return False, 0.0
        if any(kw in lower for kw in self.strong_keywords):
            return True, 1.0
        weak_hit = any(kw in lower for kw in self.weak_keywords)
        if weak_hit:
            if any(chat in lower for chat in self.chat_indicators):
                return True, 0.4
            return True, 0.7
        if len(text.split()) > 3 and re.search(r'[a-zA-Z]{4,}', text):
            if re.search(r'(review|literature|mechanism|regulation|pathway|effect|role|function|analysis|model|algorithm|system|design)', lower):
                return True, 0.6
            return True, 0.3
        return False, 0.0

    async def _llm_judge(self, text: str) -> bool:
        try:
            prompt = f"""判断用户输入是否意图进行学术文献检索（例如查找文献、写综述、了解研究进展）。
用户输入：{text}
只回答“是”或“否”。"""
            full_response = ""
            async for chunk in self.chat_agent.chat_stream([{"role": "user", "content": prompt}]):
                full_response += chunk
            if "是" in full_response:
                return True
            elif "否" in full_response:
                return False
            else:
                return True
        except Exception as e:
            print(f"LLM judge failed: {e}, fallback to rule")
            return self._is_search_intent(text)[0]

    def _is_greeting(self, text: str) -> bool:
        lower = text.lower().strip()
        return any(g in lower for g in self.greeting_patterns)

    # ---------- 保留原有 _summarize_text（用于专家模式综述） ----------
    async def _summarize_text(self, text: str, user_query: str) -> str:
        """调用 LLM 对长文本进行详细总结（结构化综述，更丰富的内容）"""
        if len(text) > 20000:
            text = text[:20000] + "...(截断)"
        
        prompt = f"""请根据以下文献内容，写一份详实、结构清晰的文献综述（按以下五个部分），总字数控制在 800-1000 字，每个部分至少包含 3-5 个要点，适当展开论述。

用户查询：{user_query}

内容：
{text}

综述结构要求：
## 1. 研究背景与意义
...
## 2. 方法与技术路线
...
## 3. 主要发现与结果
...
## 4. 讨论与综合分析
...
## 5. 结论与展望
...
请使用正式、学术的表述，确保逻辑连贯，内容充实。"""
        
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.chat_agent.client.chat.completions.create(
                model=self.chat_agent.model,
                messages=messages,
                temperature=0.5,
                max_tokens=2500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"总结失败: {e}"

    # ---------- 核心 WebSocket 处理 ----------
    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        session_id = id(websocket)
        # 初始化文献上下文（空）
        self.session_docs[session_id] = None
        # 初始化检索结果（空）
        self.session_search_results[session_id] = None

        try:
            while True:
                message = await websocket.receive_text()
                try:
                    data = json.loads(message)
                    user_msg = data.get("content", "").strip()
                    # 如果有模式切换，但已废弃，可忽略
                except:
                    user_msg = message.strip()
                if not user_msg:
                    continue

                lower_msg = user_msg.lower()

                # ----- 清除文献上下文指令 -----
                if "清除文献" in lower_msg or "退出文献" in lower_msg:
                    self.session_docs[session_id] = None
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "System",
                        "content": "🧹 已清除当前文献上下文，您可以上传新的文献或进行其他操作。"
                    }, default=self.serialize_object))
                    await websocket.send_text(json.dumps({"type": "end"}, default=self.serialize_object))
                    continue

                # ----- 清除检索结果指令 -----
                if "清除检索" in lower_msg or "退出检索" in lower_msg:
                    self.session_search_results[session_id] = None
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "System",
                        "content": "🧹 已清除检索结果，退出检索交互模式。"
                    }, default=self.serialize_object))
                    await websocket.send_text(json.dumps({"type": "end"}, default=self.serialize_object))
                    continue

                # ----- 文献解析指令 -----
                if user_msg.startswith("解析文件:"):
                    file_id = user_msg[len("解析文件:"):].strip()
                    text = file_cache.get(file_id)
                    if not text:
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "agent": "System",
                            "content": "文件不存在或已过期，请重新上传"
                        }, default=self.serialize_object))
                        await websocket.send_text(json.dumps({"type": "end"}, default=self.serialize_object))
                        continue
                    # 存储文献全文到会话
                    self.session_docs[session_id] = {"text": text, "history": []}
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "Thinking",
                        "content": "📄 正在生成文献卡片..."
                    }, default=self.serialize_object))
                    card_md = await self._extract_card(text, user_msg)
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "FastSummary",
                        "content": card_md
                    }, default=self.serialize_object))
                    await websocket.send_text(json.dumps({"type": "end"}, default=self.serialize_object))
                    continue

                if user_msg.startswith("解析URL:"):
                    url = user_msg[len("解析URL:"):].strip()
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "Thinking",
                        "content": f"🌐 正在获取 URL 内容: {url}"
                    }, default=self.serialize_object))
                    try:
                        text = parse_url(url)
                        if "失败" in text:
                            await websocket.send_text(json.dumps({
                                "type": "update",
                                "agent": "System",
                                "content": f"获取 URL 失败: {text}"
                            }, default=self.serialize_object))
                            await websocket.send_text(json.dumps({"type": "end"}, default=self.serialize_object))
                            continue
                        # 存储文献全文到会话
                        self.session_docs[session_id] = {"text": text, "history": []}
                        card_md = await self._extract_card(text, user_msg)
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "agent": "FastSummary",
                            "content": card_md
                        }, default=self.serialize_object))
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "agent": "System",
                            "content": f"解析 URL 异常: {e}"
                        }, default=self.serialize_object))
                    await websocket.send_text(json.dumps({"type": "end"}, default=self.serialize_object))
                    continue

                # ----- 正常处理用户查询 -----

                # 1. 最高优先级：语义检索（匹配关键词）
                if any(kw in user_msg.lower() for kw in self.quick_search_keywords):
                    # 执行检索
                    result = await self.quick_search_agent.search(user_msg)
                    # 获取检索到的论文列表
                    papers = self.quick_search_agent.get_last_papers()
                    if papers:
                        self.session_search_results[session_id] = {
                            "papers": papers,
                            "query": user_msg
                        }
                    # 发送结果
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "QuickSearch",
                        "content": result
                    }, default=self.serialize_object))
                    await websocket.send_text(json.dumps({"type": "end"}, default=self.serialize_object))
                    continue

                # 2. 如果存在检索结果，进入检索交互模式
                if self.session_search_results.get(session_id) is not None:
                    # 用户正在针对检索结果进行追问
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "Thinking",
                        "content": "💬 正在基于检索结果回答..."
                    }, default=self.serialize_object))
                    papers = self.session_search_results[session_id]["papers"]
                    answer = await self.quick_search_agent.ask_with_results(
                        user_msg,
                        papers
                    )
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "ChatAgent",
                        "content": answer
                    }, default=self.serialize_object))
                    await websocket.send_text(json.dumps({"type": "end"}, default=self.serialize_object))
                    continue

                # 3. 文献问答模式（如果存在文献上下文）
                if self.session_docs.get(session_id) is not None:
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "Thinking",
                        "content": "📖 正在基于当前文献回答..."
                    }, default=self.serialize_object))
                    full_response = ""
                    async for chunk in self._literature_chat(session_id, user_msg):
                        full_response += chunk
                    if self.session_docs[session_id] is not None:
                        self.session_docs[session_id]["history"].append({"role": "assistant", "content": full_response})
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "ChatAgent",
                        "content": full_response
                    }, default=self.serialize_object))
                    await websocket.send_text(json.dumps({"type": "end"}, default=self.serialize_object))
                    continue

                # 4. 原有意图识别（综述等）
                is_search, confidence = self._is_search_intent(user_msg)
                if 0.3 < confidence < 0.8:
                    is_search = await self._llm_judge(user_msg)
                elif confidence <= 0.3:
                    is_search = False

                if is_search:
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "Thinking",
                        "content": "🤔 正在思考..."
                    }, default=self.serialize_object))
                    async for update in self.search_agent.process_query(user_msg):
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "agent": update.get("agent"),
                            "content": update.get("content")
                        }, default=self.serialize_object))
                else:
                    # 普通问答
                    if self._is_greeting(user_msg):
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "agent": "ChatAgent",
                            "content": "🤖 正在思考...\n\n"
                        }, default=self.serialize_object))
                        reply = "你好！我是你的文献综述助手，可以帮助你检索和总结学术文献(目前仅限开源文献)。有什么需要我帮忙的吗？"
                        for ch in reply:
                            await websocket.send_text(json.dumps({
                                "type": "update",
                                "agent": "ChatAgent",
                                "content": ch
                            }, default=self.serialize_object))
                            await asyncio.sleep(0.02)
                    elif any(kw in user_msg.lower() for kw in self.function_keywords):
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "agent": "ChatAgent",
                            "content": "🤖 正在思考...\n\n"
                        }, default=self.serialize_object))
                        reply = "我的主要功能包括：\n\n1. **语义检索**：基于 Exa 搜索引擎，精确查找相关文献。\n2. **文献解析**：上传文件或输入 URL，自动提取内容并生成综述。\n3. **领域总结**：生成文献综述，梳理研究进展。"
                        for ch in reply:
                            await websocket.send_text(json.dumps({
                                "type": "update",
                                "agent": "ChatAgent",
                                "content": ch
                            }, default=self.serialize_object))
                            await asyncio.sleep(0.02)
                    else:
                        messages = [
                            {"role": "system", "content": "你是一个文献综述助手，专门帮助用户查找和总结学术文献。你的名字是 PAPERSAGE 助手。请不要介绍自己为 DeepSeek 或其他公司。当用户问候时，友好地介绍自己的功能。"},
                            {"role": "user", "content": user_msg}
                        ]
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "agent": "ChatAgent",
                            "content": "🤖 正在思考...\n\n"
                        }, default=self.serialize_object))
                        async for chunk in self.chat_agent.chat_stream(messages):
                            await websocket.send_text(json.dumps({
                                "type": "update",
                                "agent": "ChatAgent",
                                "content": chunk
                            }, default=self.serialize_object))

                # 结束标记
                await websocket.send_text(json.dumps({"type": "end"}, default=self.serialize_object))
        except Exception as e:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}, default=self.serialize_object))
        finally:
            # 清理会话资源
            if session_id in self.session_docs:
                del self.session_docs[session_id]
            if session_id in self.session_search_results:
                del self.session_search_results[session_id]

    def serialize_object(self, obj):
        if hasattr(obj, "__class__") and obj.__class__.__name__ == "FunctionCall":
            return {
                "id": getattr(obj, "id", None),
                "function": getattr(obj, "function", None),
                "arguments": getattr(obj, "arguments", None),
            }
        if hasattr(obj, "__dict__"):
            return {k: self.serialize_object(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        if isinstance(obj, (set, tuple)):
            return list(obj)
        try:
            return str(obj)
        except:
            return f"Unserializable object of type {type(obj).__name__}"