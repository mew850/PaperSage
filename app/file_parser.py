import requests
from bs4 import BeautifulSoup
import PyPDF2
import pdfplumber
import io

def parse_pdf(file_path: str) -> str:
    """提取 PDF 文本（优先 pdfplumber，回退 PyPDF2）"""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''.join(page.extract_text() or '' for page in pdf.pages)
        if text.strip():
            return text
    except:
        pass
    # 回退
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''.join(page.extract_text() or '' for page in reader.pages)
    return text

def parse_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def parse_url(url: str) -> str:
    """从 URL 获取文本内容（HTML 或纯文本）"""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '').lower()
        if 'pdf' in content_type:
            # 下载 PDF 并解析
            with io.BytesIO(resp.content) as pdf_io:
                with pdfplumber.open(pdf_io) as pdf:
                    text = ''.join(page.extract_text() or '' for page in pdf.pages)
            return text
        else:
            # 解析 HTML
            soup = BeautifulSoup(resp.text, 'html.parser')
            # 移除 script 和 style
            for s in soup(['script', 'style']):
                s.decompose()
            return soup.get_text(separator='\n')
    except Exception as e:
        return f"获取 URL 内容失败: {e}"