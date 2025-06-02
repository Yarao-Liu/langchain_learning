# 导入必要的库
from langchain_community.document_loaders import TextLoader  # 用于加载文本文件
from langchain_text_splitters import MarkdownHeaderTextSplitter  # Markdown文档分割器

# 加载Markdown文件
# 使用TextLoader加载README.md文件，指定UTF-8编码以正确处理中文
loader = TextLoader("../README.md",encoding="utf-8")
doc = loader.load()
print(doc)

# 定义Markdown标题分割规则
# 每个元组包含两个元素：
# 1. 标题标记（如"#"、"##"等）
# 2. 对应的标题级别名称
headers_to_split_on = [
    ("#", "Header 1"),    # 一级标题，对应Markdown中的"# 标题"
    ("##", "Header 2"),   # 二级标题，对应Markdown中的"## 标题"
    ("###", "Header 3")   # 三级标题，对应Markdown中的"### 标题"
]

# 创建Markdown分割器实例
# 使用上面定义的标题规则来分割文档
markdown_spliter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
)

# 执行文档分割
# 将文档内容按照标题层级进行分割
markdown_splits = markdown_spliter.split_text(doc[0].page_content)

# 打印分割结果
# 每个分割后的部分都包含其对应的标题层级信息
print(markdown_splits)