import json
import time
import asyncio
import os
import re
import tempfile
import shutil
from typing import Optional
from fastapi import FastAPI, WebSocket, Request, Header, HTTPException, UploadFile, File as FastAPIFile
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# 从 chat_handler 导入必要的类和 file_cache（统一缓存）
from .chat_handler import ChatHandler, file_cache
from .file_parser import parse_pdf, parse_txt
from .agents import AcademicSearchTeam, SimpleChatAgent

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化 ChatHandler（用于 WebSocket）
chat_handler = ChatHandler()

# ---------- 清小搭兼容端点 ----------
VALID_API_KEY = os.getenv("LITERAS_API_KEY", "sk-your-secret-key")

def check_auth(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing credential")
    token = authorization[len("Bearer "):]
    if token != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid credential")

# 全局 Agent 实例（用于 /v1/chat/completions）
search_agent = AcademicSearchTeam()
chat_agent = SimpleChatAgent()

@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    return {"object": "list", "data": [{"id": "literas-default", "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    body = await request.json()
    stream = bool(body.get("stream", False))
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens")

    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            user_msg = m["content"].strip()
            break
    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message")

    # ----- 快速路径：探测请求（max_tokens <= 10）-----
    if max_tokens is not None and max_tokens <= 10:
        reply = "OK"
        cid = f"chatcmpl-{int(time.time()*1000)}"
        created = int(time.time())
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if stream:
            async def fast_sse():
                payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                           'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}
                yield f"data: {json.dumps(payload)}\n\n"
                payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                           'choices': [{'index': 0, 'delta': {'content': reply}, 'finish_reason': None}]}
                yield f"data: {json.dumps(payload)}\n\n"
                payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                           'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': usage}
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(fast_sse(), media_type="text/event-stream")
        else:
            return JSONResponse({
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                "usage": usage
            })

    # ----- 意图识别（简化）-----
    search_keywords = ["综述", "文献", "研究进展", "机制", "调控", "review", "literature", "mechanism", "sleep", "drosophila"]
    is_search = any(kw in user_msg.lower() for kw in search_keywords)

    cid = f"chatcmpl-{int(time.time()*1000)}"
    created = int(time.time())

    if is_search:
        # ---------- 文献检索 + 综述（流式输出，避免超时）----------
        async def generate_search():
            # 1. 角色
            payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                       'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}
            yield f"data: {json.dumps(payload)}\n\n"
            # 2. 准备提示
            payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                       'choices': [{'index': 0, 'delta': {'content': '📚 正在准备文献检索...\n'}, 'finish_reason': None}]}
            yield f"data: {json.dumps(payload)}\n\n"

            # 3. 迭代 process_query
            async for update in search_agent.process_query(user_msg):
                agent = update.get("agent")
                content = update.get("content", "")
                if not content:
                    continue

                if agent == "QueryPlanner":
                    payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                               'choices': [{'index': 0, 'delta': {'content': '📋 正在优化搜索策略...\n'}, 'finish_reason': None}]}
                    yield f"data: {json.dumps(payload)}\n\n"
                elif agent == "SearchAgent":
                    if "Successfully retrieved" in content:
                        match = re.search(r"retrieved (\d+) articles", content)
                        if match:
                            msg = f"🔍 已检索到 {match.group(1)} 篇文献\n"
                        else:
                            msg = "🔍 正在检索文献...\n"
                    else:
                        msg = "🔍 正在检索文献...\n"
                    payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                               'choices': [{'index': 0, 'delta': {'content': msg}, 'finish_reason': None}]}
                    yield f"data: {json.dumps(payload)}\n\n"
                elif agent == "SynthesisAgent":
                    payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                               'choices': [{'index': 0, 'delta': {'content': '✍️ 正在撰写综述...\n\n'}, 'finish_reason': None}]}
                    yield f"data: {json.dumps(payload)}\n\n"
                    chunks = content.split('\n')
                    for chunk in chunks:
                        if chunk.strip():
                            payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                                       'choices': [{'index': 0, 'delta': {'content': chunk + '\n'}, 'finish_reason': None}]}
                            yield f"data: {json.dumps(payload)}\n\n"
                            await asyncio.sleep(0.02)
                elif agent == "FormatterAgent":
                    payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                               'choices': [{'index': 0, 'delta': {'content': '📝 最终格式化...\n\n'}, 'finish_reason': None}]}
                    yield f"data: {json.dumps(payload)}\n\n"
                    chunks = content.split('\n')
                    for chunk in chunks:
                        if chunk.strip():
                            payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                                       'choices': [{'index': 0, 'delta': {'content': chunk + '\n'}, 'finish_reason': None}]}
                            yield f"data: {json.dumps(payload)}\n\n"
                            await asyncio.sleep(0.02)
                else:
                    pass

            usage = {"prompt_tokens": len(user_msg)//4, "completion_tokens": 0, "total_tokens": 0}
            payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                       'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': usage}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        if stream:
            return StreamingResponse(generate_search(), media_type="text/event-stream")
        else:
            final_content = ""
            async for update in search_agent.process_query(user_msg):
                if update.get("agent") in ["SynthesisAgent", "FormatterAgent"]:
                    final_content += update.get("content", "")
            final_content = final_content.replace("**TERMINATE**", "").strip()
            return JSONResponse({
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": final_content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(user_msg)//4, "completion_tokens": len(final_content)//4, "total_tokens": (len(user_msg)+len(final_content))//4}
            })
    else:
        # ---------- 普通对话（流式或非流式）----------
        async def generate_chat():
            payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                       'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}
            yield f"data: {json.dumps(payload)}\n\n"
            full_content = ""
            async for chunk in chat_agent.chat_stream([{"role": "user", "content": user_msg}]):
                full_content += chunk
                payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                           'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]}
                yield f"data: {json.dumps(payload)}\n\n"
            usage = {"prompt_tokens": len(user_msg)//4, "completion_tokens": len(full_content)//4, "total_tokens": (len(user_msg)+len(full_content))//4}
            payload = {'id': cid, 'object': 'chat.completion.chunk', 'created': created,
                       'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': usage}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        if stream:
            return StreamingResponse(generate_chat(), media_type="text/event-stream")
        else:
            full_content = ""
            async for chunk in chat_agent.chat_stream([{"role": "user", "content": user_msg}]):
                full_content += chunk
            usage = {"prompt_tokens": len(user_msg)//4, "completion_tokens": len(full_content)//4, "total_tokens": (len(user_msg)+len(full_content))//4}
            return JSONResponse({
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": full_content}, "finish_reason": "stop"}],
                "usage": usage
            })

# ---------- 文件上传端点（使用统一的 file_cache） ----------
@app.post("/upload")
async def upload_file(file: UploadFile = FastAPIFile(...)):
    """上传本地文件（PDF/TXT），返回 file_id 供后续解析"""
    file_id = f"file_{int(time.time()*1000)}"
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    if ext == '.pdf':
        text = parse_pdf(tmp_path)
    elif ext == '.txt':
        text = parse_txt(tmp_path)
    else:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail="仅支持 PDF 或 TXT 文件")
    os.unlink(tmp_path)
    # 使用从 chat_handler 导入的 file_cache
    file_cache[file_id] = text
    return {"file_id": file_id, "filename": file.filename, "preview": text[:200] + "..."}

# ---------- WebSocket 和根路径 ----------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await chat_handler.handle_websocket(websocket)

@app.get("/")
async def read_root():
    return {"status": "Literature Review Agent is running"}

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/guide.svg")