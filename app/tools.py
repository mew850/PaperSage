import aiohttp
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import asyncio
import os
import urllib.parse

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