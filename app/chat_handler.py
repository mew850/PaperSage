import json
import re
import asyncio
from fastapi import WebSocket
from .agents import AcademicSearchTeam, SimpleChatAgent, SemanticSearchAgent

class ChatHandler:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.search_agent = AcademicSearchTeam(api_key=api_key, base_url=base_url, model=model)
        self.chat_agent = SimpleChatAgent(api_key=api_key, base_url=base_url, model=model)
        self.quick_search_agent = SemanticSearchAgent(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        
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
            "最新新闻", "latest news", "trending", "关于", "about"
        ]
        # 功能询问关键词
        self.function_keywords = [
            "功能", "能做什么", "有什么功能", "可以做什么", "能力", "capabilities",
            "what can you do", "functions"
        ]

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

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_text()
                try:
                    data = json.loads(message)
                    user_msg = data.get("content", "").strip()
                except:
                    user_msg = message.strip()
                if not user_msg:
                    continue

                # 最高优先级：快速语义检索
                if any(kw in user_msg.lower() for kw in self.quick_search_keywords):
                    table = await self.quick_search_agent.search(user_msg)
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "QuickSearch",
                        "content": table
                    }, default=self.serialize_object))
                    await websocket.send_text(json.dumps({"type": "end"}, default=self.serialize_object))
                    continue

                # 综述检索 vs 普通问答
                is_search, confidence = self._is_search_intent(user_msg)
                if 0.3 < confidence < 0.8:
                    is_search = await self._llm_judge(user_msg)
                elif confidence <= 0.3:
                    is_search = False

                if is_search:
                    # 发送“正在思考”状态（消除空白）
                    await websocket.send_text(json.dumps({
                        "type": "update",
                        "agent": "Thinking",
                        "content": "🤔 正在思考..."
                    }, default=self.serialize_object))
                    # 执行综述检索
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
                        # 功能询问直接回复
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "agent": "ChatAgent",
                            "content": "🤖 正在思考...\n\n"
                        }, default=self.serialize_object))
                        reply = "我的主要功能包括：\n\n1. **语义检索**：基于 Exa 搜索引擎，精确查找相关文献。\n2. **文献解析**：阅读并提炼文献核心内容。\n3. **领域总结**：生成文献综述，梳理研究进展。"
                        for ch in reply:
                            await websocket.send_text(json.dumps({
                                "type": "update",
                                "agent": "ChatAgent",
                                "content": ch
                            }, default=self.serialize_object))
                            await asyncio.sleep(0.02)
                    else:
                        messages = [
                            {"role": "system", "content": "你是一个文献综述助手，专门帮助用户查找和总结学术文献。你的名字是 LITERAS 助手。请不要介绍自己为 DeepSeek 或其他公司。当用户问候时，友好地介绍自己的功能。"},
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