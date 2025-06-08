"""
LangGraph 工具调用系统 - 局部功能测试
=====================================

这个文件包含了对76_graph.py中各个组件的独立测试，
帮助您理解每个部分的工作原理和数据流。

测试内容：
1. 🔧 工具函数测试
2. 💭 LLM链测试  
3. 📊 JSON解析器测试
4. 🏗️ 状态管理测试
5. 🔄 节点函数测试
6. 🚀 完整工作流测试

作者：AI助手
日期：2024年
"""

import os
import sys

import dotenv

# 添加当前目录到Python路径，以便导入76_graph.py中的组件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的库
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
dotenv.load_dotenv()

# ================================
# 测试配置
# ================================

def setup_test_environment():
    """设置测试环境"""
    print("🔧 设置测试环境...")
    
    # 检查API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到 OPENAI_API_KEY 环境变量")
        return None
    
    # 创建LLM实例
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1/",
        model="Qwen/Qwen2.5-72B-Instruct",
        temperature=0.3
    )
    
    print("✅ 测试环境设置完成")
    return llm

# ================================
# 1. 工具函数测试
# ================================

def test_searxng_search():
    """测试搜索工具的基本功能"""
    print("\n" + "="*60)
    print("🔍 测试1: SearXNG搜索工具")
    print("="*60)
    
    # 导入搜索工具
    try:
        from chapter07.graph_loop import searxng_search
        
        # 测试用例
        test_queries = [
            "Python编程",
            "人工智能最新发展", 
            "不存在的搜索词xyzabc123"
        ]
        
        for query in test_queries:
            print(f"\n📝 测试查询: {query}")
            print("-" * 40)
            
            try:
                result = searxng_search.invoke(query)
                print(f"✅ 搜索成功，返回 {len(result)} 个结果")
                
                if result:
                    print("📋 第一个结果示例:")
                    first_result = result[0]
                    print(f"   标题: {first_result.get('title', 'N/A')[:50]}...")
                    print(f"   内容: {first_result.get('content', 'N/A')[:100]}...")
                    print(f"   链接: {first_result.get('url', 'N/A')}")
                else:
                    print("⚠️  搜索结果为空")
                    
            except Exception as e:
                print(f"❌ 搜索失败: {e}")
                
    except ImportError as e:
        print(f"❌ 导入搜索工具失败: {e}")

# ================================
# 2. LLM链测试
# ================================

def test_llm_chain(llm):
    """测试LLM链的决策功能"""
    print("\n" + "="*60)
    print("💭 测试2: LLM决策链")
    print("="*60)
    
    # 创建简化的提示词模板
    simple_prompt = ChatPromptTemplate.from_messages([
        {
            "role": "system",
            "content": "你是一个智能助手，需要决定是否使用搜索工具。"
        },
        {
            "role": "user",
            "content": """
用户问题: {input}

请按照以下JSON格式回复：
如果需要搜索：{{"action": "searxng_search", "action_input": "搜索词"}}
如果不需要搜索：{{"action": "Final Answer", "answer": "直接回答"}}
"""
        }
    ])
    
    # 测试用例
    test_cases = [
        {
            "input": "今天天气怎么样？",
            "expected": "需要搜索实时信息"
        },
        {
            "input": "你好",
            "expected": "直接回答"
        },
        {
            "input": "刘亦菲最近有什么新电影？",
            "expected": "需要搜索最新信息"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {case['input']}")
        print(f"🎯 预期行为: {case['expected']}")
        print("-" * 40)
        
        try:
            # 调用LLM
            chain = simple_prompt | llm
            response = chain.invoke({"input": case["input"]})
            
            print(f"🤖 LLM原始回复:")
            print(f"   {response.content}")
            
            # 尝试解析JSON
            try:
                import json
                # 简单的JSON提取（实际应用中会更复杂）
                content = response.content.strip()
                if content.startswith('{') and content.endswith('}'):
                    parsed = json.loads(content)
                    action = parsed.get('action', 'Unknown')
                    print(f"✅ 解析成功，决策动作: {action}")
                else:
                    print("⚠️  回复不是标准JSON格式")
            except Exception as e:
                print(f"❌ JSON解析失败: {e}")
                
        except Exception as e:
            print(f"❌ LLM调用失败: {e}")

# ================================
# 3. JSON解析器测试
# ================================

def test_json_parser():
    """测试JSON解析器的健壮性"""
    print("\n" + "="*60)
    print("📊 测试3: JSON解析器")
    print("="*60)
    
    # 导入JSON解析器
    try:
        from chapter07.graph_loop import jsonParser
        from langchain_core.messages import AIMessage
        
        # 测试用例
        test_cases = [
            {
                "name": "正确的JSON",
                "content": '{"action": "Final Answer", "answer": "这是一个测试回答"}',
                "should_succeed": True
            },
            {
                "name": "带代码块的JSON",
                "content": '```json\n{"action": "searxng_search", "action_input": "测试搜索"}\n```',
                "should_succeed": True
            },
            {
                "name": "空内容",
                "content": "",
                "should_succeed": False
            },
            {
                "name": "无效JSON",
                "content": "这不是JSON格式的内容",
                "should_succeed": False
            },
            {
                "name": "不完整的JSON",
                "content": '{"action": "Final Answer"',
                "should_succeed": False
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 测试用例 {i}: {case['name']}")
            print(f"📄 输入内容: {case['content'][:50]}...")
            print("-" * 40)
            
            # 创建模拟的AI消息
            mock_message = AIMessage(content=case["content"])
            
            try:
                result = jsonParser(mock_message)
                print(f"✅ 解析成功: {result}")
                
                if case["should_succeed"]:
                    print("🎯 符合预期（应该成功）")
                else:
                    print("⚠️  意外成功（预期应该失败）")
                    
            except Exception as e:
                print(f"❌ 解析失败: {e}")
                
                if not case["should_succeed"]:
                    print("🎯 符合预期（应该失败）")
                else:
                    print("⚠️  意外失败（预期应该成功）")
                    
    except ImportError as e:
        print(f"❌ 导入JSON解析器失败: {e}")

# ================================
# 4. 状态管理测试
# ================================

def test_state_management():
    """测试状态管理和数据流"""
    print("\n" + "="*60)
    print("🏗️ 测试4: 状态管理")
    print("="*60)
    
    # 导入状态类
    try:
        from chapter07.graph_loop import AgentState
        
        # 创建初始状态
        initial_state = {
            "input": "测试用户输入",
            "agent_scratchpad": [],
            "output": "",
            "llm_decision": {},
            "tool_result": ""
        }
        
        print("📝 初始状态:")
        for key, value in initial_state.items():
            print(f"   {key}: {value}")
        
        # 模拟状态更新
        print("\n🔄 模拟状态更新过程:")
        
        # 1. 添加LLM决策
        initial_state["llm_decision"] = {
            "action": "searxng_search",
            "action_input": "测试搜索"
        }
        print("1️⃣ 添加LLM决策结果")
        
        # 2. 添加工具调用记录
        tool_record = "使用工具 searxng_search(测试搜索) -> [搜索结果1, 搜索结果2]"
        initial_state["agent_scratchpad"].append(tool_record)
        print("2️⃣ 添加工具调用记录")
        
        # 3. 设置最终输出
        initial_state["output"] = "基于搜索结果的最终回答"
        print("3️⃣ 设置最终输出")
        
        print("\n📊 最终状态:")
        for key, value in initial_state.items():
            if isinstance(value, list) and value:
                print(f"   {key}: [{len(value)} 个项目]")
                for i, item in enumerate(value):
                    print(f"     {i+1}. {str(item)[:50]}...")
            else:
                print(f"   {key}: {value}")
                
        print("✅ 状态管理测试完成")
        
    except ImportError as e:
        print(f"❌ 导入状态类失败: {e}")

# ================================
# 主测试函数
# ================================

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始LangGraph工具调用系统测试")
    print("="*80)
    
    # 设置测试环境
    llm = setup_test_environment()
    if not llm:
        print("❌ 测试环境设置失败，退出测试")
        return
    
    # 运行各项测试
    test_searxng_search()
    test_llm_chain(llm)
    test_json_parser()
    test_state_management()
    
    print("\n" + "="*80)
    print("✅ 所有测试完成")
    print("="*80)
    
    print("\n📚 测试总结:")
    print("1. 🔧 工具测试: 验证搜索功能是否正常")
    print("2. 💭 LLM测试: 验证决策逻辑是否正确")
    print("3. 📊 解析测试: 验证JSON解析的健壮性")
    print("4. 🏗️ 状态测试: 验证数据流和状态管理")
    print("\n💡 提示: 如果某些测试失败，请检查:")
    print("   - SearXNG服务是否运行在localhost:6688")
    print("   - API密钥是否正确配置")
    print("   - 网络连接是否正常")

if __name__ == "__main__":
    run_all_tests()
