"""
Exa 接口测试
"""

from exa_py import Exa

if __name__ == "__main__":
    exa = Exa("715b0efd-b662-4e90-b7ac-c721e59052d9")

    # 执行搜索
    result = exa.search(
        "halide solid state electrolyte",  # 搜索查询
        num_results=1,            # 返回结果数量
        type="auto",              # 自动选择搜索模式（instant / fast / auto / deep）
        contents={                # 返回内容配置
            "highlights": True    # 返回高亮片段
        }
    )

    print(result)