# 导入必要的库
import time  # 用于计算执行时间
import os    # 用于处理文件路径和目录操作
import sqlite3  # 用于SQLite数据库操作

# 导入LangChain相关组件
from langchain_community.cache import SQLiteCache  # SQLite缓存实现
from langchain_core.globals import set_llm_cache   # 设置全局缓存
from langchain_ollama import OllamaLLM             # Ollama大语言模型接口

# 初始化Ollama大语言模型
# 使用Qwen 2.5 7B模型，这是一个支持中文的模型
# base_url指定Ollama服务的地址，默认是本地服务
llm = OllamaLLM(
    model="qwen2.5:7b",  # 使用Qwen 2.5 7B模型
    base_url="http://localhost:11434",  # Ollama服务地址，可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 定义测试问题
question = "请问两只兔子有多少条腿?"

# 第一次测试：不使用缓存
# 这个测试会直接调用模型，不经过缓存
print("不使用缓存的测试：")
start_time = time.time()  # 记录开始时间
response1 = llm.invoke(question)  # 直接调用模型
end_time = time.time()    # 记录结束时间
print(f"回答: {response1}")
print(f"耗时: {end_time - start_time:.2f} 秒\n")  # 计算并显示耗时

# 设置SQLite缓存
# 首先确保数据库目录存在
db_dir = os.path.join(os.path.dirname(__file__), "db")  # 获取当前脚本所在目录下的db文件夹路径
if not os.path.exists(db_dir):
    os.makedirs(db_dir)  # 如果目录不存在，创建它
    print(f"创建数据库目录: {db_dir}")

# 设置数据库文件路径
db_path = os.path.join(db_dir, "langchain.db")
print(f"数据库路径: {db_path}")

# 测试数据库连接
# 这一步确保我们可以正常创建和访问数据库文件
try:
    conn = sqlite3.connect(db_path)  # 尝试连接数据库
    conn.close()  # 关闭连接
    print("数据库连接测试成功")
except Exception as e:
    print(f"数据库连接测试失败: {e}")
    raise  # 如果连接失败，抛出异常

# 初始化并设置SQLite缓存
try:
    cache = SQLiteCache(database_path=db_path)  # 创建SQLite缓存实例
    set_llm_cache(cache)  # 设置为全局缓存
    print("SQLite缓存设置成功")
except Exception as e:
    print(f"设置SQLite缓存失败: {e}")
    raise  # 如果设置失败，抛出异常

# 第二次测试：使用缓存
# 这次调用会先检查缓存，如果缓存中没有，才会调用模型
print("\n使用缓存的测试：")
start_time = time.time()
try:
    response2 = llm.invoke(question)  # 调用模型，结果会被缓存
    end_time = time.time()
    print(f"回答: {response2}")
    print(f"耗时: {end_time - start_time:.2f} 秒\n")
except Exception as e:
    print(f"调用LLM时出错: {e}")

# 第三次测试：再次使用缓存
# 这次调用应该会直接从缓存中获取结果，速度会更快
print("再次使用缓存的测试：")
start_time = time.time()
try:
    response3 = llm.invoke(question)  # 这次应该会直接从缓存获取结果
    end_time = time.time()
    print(f"回答: {response3}")
    print(f"耗时: {end_time - start_time:.2f} 秒")
except Exception as e:
    print(f"调用LLM时出错: {e}")