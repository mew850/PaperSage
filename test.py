import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ELSEVIER_API_KEY")

def parse_elsevier_abstract(response_dict):
    """
    从 Elsevier Abstract Retrieval API 的响应中提取关键信息
    """
    # 进入最外层
    root = response_dict.get('abstracts-retrieval-response', {})
    if not root:
        return None

    # 进入 item 和 bibrecord
    item = root.get('item', {})
    bibrecord = item.get('bibrecord', {})
    head = bibrecord.get('head', {})

    # 1. 标题
    title = head.get('citation-title', '无标题')

    # 2. 摘要
    abstract = head.get('abstracts', '无摘要')

    # 3. 期刊名称
    source = bibrecord.get('source', {})
    journal = source.get('sourcetitle', '未知期刊')

    # 4. 发表日期（可能有 publicationdate 字段）
    pub_date = source.get('publicationdate', {})
    year = pub_date.get('year', '未知年份')
    month = pub_date.get('month', '')
    day = pub_date.get('day', '')
    pub_date_str = f"{year}-{month}-{day}".strip('-')

    # 5. 作者列表
    authors = []
    author_groups = head.get('author-group', [])
    for group in author_groups:
        for author in group.get('author', []):
            given = author.get('ce:given-name', '')
            surname = author.get('ce:surname', '')
            full_name = f"{given} {surname}".strip()
            if full_name:
                authors.append(full_name)

    # 6. DOI（可以从请求的 URL 中获取，也可在响应中查找）
    # 有些响应会包含 prism:doi 字段
    doi = item.get('prism:doi', '未提供')

    # 7. 关键词（如果有）
    citation_info = head.get('citation-info', {})
    keywords = []
    kw_list = citation_info.get('author-keywords', {}).get('author-keyword', [])
    for kw in kw_list:
        if isinstance(kw, dict):
            keywords.append(kw.get('$', ''))
        else:
            keywords.append(str(kw))

    return {
        'title': title,
        'abstract': abstract,
        'journal': journal,
        'publication_date': pub_date_str,
        'authors': authors,
        'doi': doi,
        'keywords': keywords
    }


# ---------- 示例调用 ----------
if __name__ == "__main__":
    # 假设我们已经成功获取了响应数据（字典）
    # 这里重新演示一下请求过程（使用您之前的方式）
    DOI = "10.1016/j.patter.2023.100801"
    url = f"https://api.elsevier.com/content/abstract/doi/{DOI}"
    headers = {
        "X-ELS-APIKey": API_KEY,
        "Accept": "application/json"
    }

    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        parsed = parse_elsevier_abstract(data)

        # 打印结果
        print("=" * 60)
        print(f"📄 标题：{parsed['title']}")
        print(f"📚 期刊：{parsed['journal']}")
        print(f"📅 发表日期：{parsed['publication_date']}")
        print(f"🔗 DOI：{parsed['doi']}")
        print(f"👤 作者：{', '.join(parsed['authors'])}")
        print(f"🏷️  关键词：{', '.join(parsed['keywords'])}")
        print("\n📝 摘要：")
        print(parsed['abstract'][:500] + "..." if len(parsed['abstract']) > 500 else parsed['abstract'])
        print("=" * 60)
    else:
        print(f"请求失败，状态码 {resp.status_code}")
        print(resp.text[:500])
