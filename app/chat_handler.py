import json
import re
import asyncio
from fastapi import WebSocket
from .agents import AcademicSearchTeam, SimpleChatAgent

class ChatHandler:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.search_agent = AcademicSearchTeam(api_key=api_key, base_url=base_url, model=model)
        self.chat_agent = SimpleChatAgent(api_key=api_key, base_url=base_url, model=model)
        # 意图识别配置
        self.strong_keywords = [
            "综述", "文献", "研究进展", "最新研究", "机制", "调控", "通路",
            "review", "literature", "mechanism", "pathway", "regulate",
            "meta-analysis", "systematic review"
        ]
        self.weak_keywords = [
            "影响", "作用", "发现", "报道", "实验", "研究", "基因", "蛋白",
            "effect", "role", "function", "study", "experiment"
        ]
        self.negation_patterns = [
            "不要", "不需要", "不用", "别", "不想", "没", "不要综述", "不查文献"
        ]
        self.chat_indicators = [
            "你好", "hi", "hello", "谢谢", "thanks", "再见", "bye", "哈哈", "嗯嗯"
        ]
        # 问候检测（用于直接回复）
        self.greeting_patterns = [
            "你好", "hi", "hello", "hey", "你是谁", "你叫什么", "介绍一下你自己",
            "介绍一下", "你是谁呀", "你好呀"
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
            return True, 0.3
        return False, 0.0

    def _is_greeting(self, text: str) -> bool:
        lower = text.lower().strip()
        return any(g in lower for g in self.greeting_patterns)

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

                # 意图识别
                is_search, confidence = self._is_search_intent(user_msg)
                if 0.3 < confidence < 0.8:
                    is_search = await self._llm_judge(user_msg)
                elif confidence <= 0.3:
                    is_search = False

                if is_search:
                    # 文献检索 + 综述
                    async for update in self.search_agent.process_query(user_msg):
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "agent": update.get("agent"),
                            "content": update.get("content")
                        }, default=self.serialize_object))
                else:
                    # 普通问答：先检测是否简单问候
                    if self._is_greeting(user_msg):
                        # 硬编码回复（模拟流式）
                        await websocket.send_text(json.dumps({
                            "type": "update",
                            "agent": "ChatAgent",
                            "content": "🤖 正在思考...\n\n"
                        }, default=self.serialize_object))
                        reply = "你好！我是你的文献综述助手PaperSage，可以帮助你检索和总结学术文献 (**暂时局限于生物领域**)。有什么需要我帮忙的吗？"
                        for ch in reply:
                            await websocket.send_text(json.dumps({
                                "type": "update",
                                "agent": "ChatAgent",
                                "content": ch
                            }, default=self.serialize_object))
                            await asyncio.sleep(0.02)  # 模拟打字效果
                    else:
                        # 带系统提示调用 LLM
                        messages = [
                            {"role": "system", "content": "你是一个文献综述助手，专门帮助用户查找和总结学术文献。你的名字是 PaperSage 助手。当用户问候时，友好地介绍自己的功能。"},
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