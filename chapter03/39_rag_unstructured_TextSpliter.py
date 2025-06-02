# 导入文本分割器相关类
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters import Language

def demonstrate_character_splitter():
    """
    演示字符分割器的使用
    """
    # 创建字符分割器实例
    text_spliter = CharacterTextSplitter(
        separator="\n\n",  # 使用双换行符作为分隔符
        chunk_size=100,    # 每个文本块的最大字符数
        chunk_overlap=10,  # 相邻文本块之间的重叠字符数
        length_function=len,  # 用于计算文本长度的函数
        is_separator_regex=False  # 分隔符是否为正则表达式
    )
    
    # 示例文本
    sample_text = """
    这是第一段文本。
    这是第一段的继续。

    这是第二段文本。
    这是第二段的继续。

    这是第三段文本。
    这是第三段的继续。
    """
    
    # 执行文本分割
    chunks = text_spliter.split_text(sample_text)
    print("字符分割结果：")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n块 {i}:")
        print(chunk)

def demonstrate_recursive_splitter():
    """
    演示递归分割器的使用
    """
    # 创建递归分割器实例
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,    # 每个文本块的最大字符数
        chunk_overlap=20,  # 相邻文本块之间的重叠字符数
        length_function=len,  # 用于计算文本长度的函数
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]  # 分隔符优先级列表
    )
    
    # 示例文本
    sample_text = """
    这是第一段文本。这是第一段的继续！
    这是第二段文本？这是第二段的继续。
    这是第三段文本！这是第三段的继续？
    """
    
    # 执行文本分割
    chunks = recursive_splitter.split_text(sample_text)
    print("\n递归分割结果：")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n块 {i}:")
        print(chunk)

def demonstrate_language_specific_splitter():
    """
    演示特定语言分割器的使用
    """
    # 打印所有支持的语言
    print("\n支持的语言列表：")
    for lang in Language:
        print(f"- {lang.name}")
    
    # 获取JavaScript语言的分隔符
    js_separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.JS)
    print("\nJavaScript语言的分隔符：")
    print(js_separators)
    
    # 创建JavaScript专用的分割器
    js_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS,
        chunk_size=100,
        chunk_overlap=20
    )
    
    # 示例JavaScript代码
    js_code = """
    function example() {
        console.log("Hello");
        if (true) {
            return "World";
        }
    }
    """
    
    # 执行代码分割
    chunks = js_splitter.split_text(js_code)
    print("\nJavaScript代码分割结果：")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n块 {i}:")
        print(chunk)

if __name__ == "__main__":
    # 运行所有演示
    demonstrate_character_splitter()
    demonstrate_recursive_splitter()
    demonstrate_language_specific_splitter()

