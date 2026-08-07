import aiohttp
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import asyncio
import os
import urllib.parse
from exa_py import Exa

PUBMED_MAX_RESULTS = 8
ABSTRACT_MAX_CHARS = 200

def configure_pubmed(*, full: bool) -> None:
    global PUBMED_MAX_RESULTS, ABSTRACT_MAX_CHARS
    if full:
        PUBMED_MAX_RESULTS = 35
        ABSTRACT_MAX_CHARS = 0
    else:
        PUBMED_MAX_RESULTS = 8
        ABSTRACT_MAX_CHARS = 200

def _truncate(text: Optional[str], n: int) -> str:
    if not text:
        return "No abstract available"
    if n <= 0 or len(text) <= n:
        return text
    return text[: n - 1] + "…"

async def pubmed_search(query: str, max_results: int = None) -> List[Dict]:
    """
    搜索 PubMed，使用 NCBI API Key（若已设置）以提高限额。
    """
    if max_results is None:
        max_results = PUBMED_MAX_RESULTS
    else:
        max_results = min(max_results, PUBMED_MAX_RESULTS)

    # 获取 API Key（若存在）
    api_key = os.getenv("NCBI_API_KEY", "").strip()
    base_params = {
        "db": "pubmed",
        "term": query,           # aiohttp 会自动编码
        "retmax": max_results,
        "retmode": "json",
    }
    if api_key:
        base_params["api_key"] = api_key
        print(f"🔑 Using NCBI API Key (prefix: {api_key[:6]}...)")
    else:
        print("⚠️  No NCBI API Key found. Using public access (rate limited).")

    print(f"🔍 PubMed query: {query}")

    ssl_context = aiohttp.TCPConnector(ssl=False)
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    async with aiohttp.ClientSession(connector=ssl_context) as session:
        # ---------- 搜索阶段 ----------
        ids = []
        for attempt in range(3):  # 最多重试3次
            try:
                async with session.get(search_url, params=base_params) as response:
                    if response.status == 429:
                        wait = 2 ** attempt
                        print(f"Rate limit exceeded. Retrying in {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Search API error: Status {response.status}, Response: {error_text[:300]}")
                        # 如果错误包含 "API key invalid"，直接返回空并提示
                        if "invalid" in error_text.lower():
                            print("❌ NCBI API Key is invalid. Please check your key.")
                        return []
                    search_data = await response.json()
                    ids = search_data.get("esearchresult", {}).get("idlist", [])
                    print(f"Found {len(ids)} article IDs")
                    break
            except Exception as e:
                print(f"Search request exception: {e}")
                return []
        else:
            print("All search attempts failed.")
            return []

        if not ids:
            print("No articles found")
            return []

        # ---------- 获取详情 ----------
        results = []
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(batch_ids),
                "retmode": "xml",
            }
            if api_key:
                fetch_params["api_key"] = api_key

            for attempt in range(3):
                try:
                    await asyncio.sleep(0.34)  # 避免过快
                    async with session.get(fetch_url, params=fetch_params) as response:
                        if response.status == 429:
                            wait = 2 ** attempt
                            print(f"Rate limit on fetch. Retrying in {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"Fetch API error: {response.status}, Response: {error_text[:200]}")
                            break
                        content = await response.text()
                        root = ET.fromstring(content)
                        for article in root.findall(".//PubmedArticle"):
                            try:
                                raw_abs = article.find(".//Abstract/AbstractText")
                                raw_abs_text = raw_abs.text if raw_abs is not None else None
                                article_data = {
                                    "title": article.find(".//ArticleTitle").text or "No title",
                                    "abstract": _truncate(raw_abs_text, ABSTRACT_MAX_CHARS),
                                    "journal": article.find(".//Journal/Title").text or "No journal",
                                    "date": "Unknown",
                                    "doi": article.find(".//ArticleId[@IdType='doi']").text or "No DOI",
                                    "first_author": "No author",
                                    "pmid": article.find(".//PMID").text or "",
                                }
                                pub_date = article.find(".//PubDate")
                                if pub_date is not None:
                                    year = pub_date.find("Year")
                                    month = pub_date.find("Month")
                                    year_text = year.text if year is not None else "Unknown"
                                    month_text = month.text if month is not None else "01"
                                    article_data["date"] = f"{year_text}-{month_text}"
                                authors = article.findall(".//Author")
                                if authors:
                                    last_name = authors[0].find("LastName")
                                    first_name = authors[0].find("ForeName")
                                    if last_name is not None and first_name is not None:
                                        article_data["first_author"] = f"{first_name.text} {last_name.text}"
                                        if len(authors) > 1:
                                            article_data["first_author"] += " et al."
                                results.append(article_data)
                            except Exception as e:
                                print(f"Error processing article: {e}")
                                continue
                        break
                except Exception as e:
                    print(f"Fetch exception: {e}")
                    break

        print(f"Successfully retrieved {len(results)} articles")
        return results

async def exa_search(query: str, max_results: int = None) -> List[Dict]:
    """
    使用 Exa 搜索引擎检索学术文献（覆盖全学科）。
    返回格式与 pubmed_search 一致，以便无缝集成。
    """
    if max_results is None:
        max_results = 8  # 与 PubMed 默认一致
    api_key = os.getenv("EXA_API_KEY", "").strip()
    if not api_key:
        print("⚠️  EXA_API_KEY 未设置，跳过 Exa 搜索")
        return []
    try:
        # 同步调用 Exa（Exa 库可能为同步，但可以在异步函数中用 run_in_executor 或直接调用）
        # 假设 Exa 库支持异步，如果不支持，可以用 loop.run_in_executor
        exa = Exa(api_key)
        # 搜索学术内容，可指定 type="neural" 以获得更好的语义理解
        result = exa.search(
            query,
            num_results=max_results,
            type="auto",          # (instant, fast, auto, deep)
            contents={
                "highlights": True, # 返回高亮片段
                "summary": False,   # 也可启用摘要，但我们用高亮作为摘要
            }
        )
        papers = []
        for item in result.results:
            # 提取高亮片段作为“摘要”
            highlights = item.highlights if hasattr(item, 'highlights') else []
            abstract = highlights[0] if highlights else "No abstract available"
            # 构建类似 PubMed 的结构
            paper = {
                "title": item.title or "No title",
                "abstract": abstract[:500],  # 截断以保持一致性
                "journal": "Exa Search",      # Exa 不提供期刊，可置为固定值或从来源推断
                "date": "Unknown",
                "doi": item.url or "",        # 用 URL 作为标识
                "first_author": "Unknown",    # Exa 不直接提供作者
                "pmid": "",                  # 无 PMID
                "url": item.url,              # 额外保留链接
                "source": "exa"               # 标记来源
            }
            papers.append(paper)
        print(f"Exa 检索到 {len(papers)} 条结果")
        return papers
    except Exception as e:
        print(f"Exa 搜索出错: {e}")
        return []