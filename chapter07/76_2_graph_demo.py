"""
LangGraph 工具调用系统演示脚本
=====================================

这个脚本提供了一个交互式的演示，帮助您理解LangGraph工具调用系统的各个组件。

功能：
1. 🔧 单独测试搜索工具
2. 💭 测试LLM决策过程
3. 🔄 演示完整的工作流程
4. 📊 展示状态变化过程

使用方法：
python 76_2_graph_demo.py

作者：AI助手
日期：2024年
"""

import os
import sys

import dotenv

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
dotenv.load_dotenv()

def print_section(title: str, emoji: str = "🔹"):
    """打印章节标题"""
    print(f"\n{emoji} " + "="*60)
    print(f"{emoji} {title}")
    print(f"{emoji} " + "="*60)

def print_step(step: str, content: str = ""):
    """打印步骤信息"""
    print(f"\n📍 {step}")
    if content:
        print(f"   {content}")
    print("-" * 50)

def demo_search_tool():
    """演示搜索工具的功能"""
    print_section("搜索工具演示", "🔍")
    
    try:
        from chapter07.graph_loop import searxng_search
        
        print("这个演示将展示SearXNG搜索工具的工作原理")
        print("注意：需要本地运行SearXNG服务在6688端口")
        
        # 获取用户输入
        query = input("\n请输入搜索关键词（回车使用默认'Python编程'）: ").strip()
        if not query:
            query = "Python编程"
        
        print_step("开始搜索", f"查询: {query}")
        
        # 执行搜索
        results = searxng_search.invoke(query)
        
        print_step("搜索完成", f"找到 {len(results)} 个结果")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n📄 结果 {i}:")
                print(f"   标题: {result.get('title', 'N/A')}")
                print(f"   内容: {result.get('content', 'N/A')[:100]}...")
                print(f"   链接: {result.get('url', 'N/A')}")
        else:
            print("⚠️  没有找到搜索结果")
            
    except ImportError as e:
        print(f"❌ 导入搜索工具失败: {e}")
    except Exception as e:
        print(f"❌ 搜索演示失败: {e}")

def demo_llm_decision():
    """演示LLM决策过程"""
    print_section("LLM决策演示", "💭")
    
    try:
        from chapter07.graph_loop import llm, prompt, jsonParser
        from langchain_core.messages import HumanMessage
        
        print("这个演示将展示LLM如何决定是否使用工具")
        
        # 获取用户输入
        user_input = input("\n请输入问题（回车使用默认问题）: ").strip()
        if not user_input:
            user_input = "今天北京的天气怎么样？"
        
        print_step("用户问题", user_input)
        
        # 构建LLM输入
        llm_input = {
            "input": user_input,
            "agent_scratchpad": []
        }
        
        print_step("调用LLM", "正在分析问题并做出决策...")
        
        # 创建LLM链
        chain = prompt | llm | jsonParser
        
        # 执行LLM调用
        decision = chain.invoke(llm_input)
        
        print_step("LLM决策结果")
        print(f"   决策: {decision}")
        
        # 解释决策
        action = decision.get("action", "Unknown")
        if action == "Final Answer":
            print("   📝 LLM决定直接回答，不需要使用工具")
            print(f"   💬 回答: {decision.get('answer', 'N/A')}")
        elif action == "searxng_search":
            print("   🔧 LLM决定使用搜索工具")
            print(f"   🔍 搜索词: {decision.get('action_input', 'N/A')}")
            print(f"   💡 原因: {decision.get('reason', 'N/A')}")
        else:
            print(f"   ❓ 未知决策: {action}")
            
    except ImportError as e:
        print(f"❌ 导入LLM组件失败: {e}")
    except Exception as e:
        print(f"❌ LLM决策演示失败: {e}")

def demo_state_flow():
    """演示状态流转过程"""
    print_section("状态流转演示", "🔄")
    
    try:
        from chapter07.graph_loop import AgentState
        
        print("这个演示将展示状态在工作流中的变化过程")
        
        # 初始状态
        state = {
            "input": "刘亦菲最近有什么新电影？",
            "agent_scratchpad": [],
            "output": "",
            "llm_decision": {},
            "tool_result": ""
        }
        
        print_step("1. 初始状态")
        for key, value in state.items():
            print(f"   {key}: {value}")
        
        # 模拟LLM决策
        state["llm_decision"] = {
            "action": "searxng_search",
            "action_input": "刘亦菲 新电影 2024",
            "reason": "需要搜索最新的电影信息"
        }
        
        print_step("2. LLM决策后")
        print(f"   llm_decision: {state['llm_decision']}")
        
        # 模拟工具调用
        tool_result = [
            {"title": "刘亦菲新电影《梦华录》", "content": "刘亦菲主演的古装剧..."},
            {"title": "刘亦菲确认出演新片", "content": "据悉刘亦菲将出演..."}
        ]
        
        state["tool_result"] = str(tool_result)
        state["agent_scratchpad"].append(
            f"使用工具 searxng_search(刘亦菲 新电影 2024) -> {len(tool_result)} 个结果"
        )
        
        print_step("3. 工具调用后")
        print(f"   tool_result: {state['tool_result'][:100]}...")
        print(f"   agent_scratchpad: {state['agent_scratchpad']}")
        
        # 模拟最终回答
        state["llm_decision"] = {
            "action": "Final Answer",
            "answer": "根据搜索结果，刘亦菲最近的作品包括《梦华录》等..."
        }
        state["output"] = state["llm_decision"]["answer"]
        
        print_step("4. 最终状态")
        print(f"   output: {state['output']}")
        print(f"   工具调用历史: {len(state['agent_scratchpad'])} 条记录")
        
    except ImportError as e:
        print(f"❌ 导入状态类失败: {e}")
    except Exception as e:
        print(f"❌ 状态流转演示失败: {e}")

def demo_full_workflow():
    """演示完整的工作流程"""
    print_section("完整工作流演示", "🚀")
    
    try:
        from chapter07.graph_loop import app
        
        print("这个演示将运行完整的LangGraph工作流")
        print("注意：需要有效的API密钥和SearXNG服务")
        
        # 获取用户输入
        user_question = input("\n请输入问题（回车使用默认问题）: ").strip()
        if not user_question:
            user_question = "什么是人工智能？"
        
        # 构建输入状态
        test_input = {
            "input": user_question,
            "agent_scratchpad": [],
            "output": "",
            "llm_decision": {},
            "tool_result": ""
        }
        
        print_step("开始执行工作流", f"问题: {user_question}")
        
        # 执行工作流
        step_count = 0
        for step in app.stream(test_input):
            step_count += 1
            print(f"\n📊 步骤 {step_count}: {list(step.keys())}")
            
            # 显示关键信息
            for node_name, node_output in step.items():
                if "llm_decision" in node_output:
                    decision = node_output["llm_decision"]
                    action = decision.get("action", "Unknown")
                    print(f"   🤖 {node_name}: 决策 = {action}")
                    
                if "tool_result" in node_output and node_output["tool_result"]:
                    result = str(node_output["tool_result"])
                    print(f"   🔧 {node_name}: 工具结果 = {result[:50]}...")
                    
                if "output" in node_output and node_output["output"]:
                    output = node_output["output"]
                    print(f"   🎯 {node_name}: 最终回答 = {output[:50]}...")
        
        # 获取最终结果
        final_result = app.invoke(test_input)
        
        print_step("工作流完成")
        print(f"   最终回答: {final_result.get('output', 'N/A')}")
        print(f"   工具调用次数: {len(final_result.get('agent_scratchpad', []))}")
        
    except ImportError as e:
        print(f"❌ 导入工作流失败: {e}")
    except Exception as e:
        print(f"❌ 完整工作流演示失败: {e}")

def main():
    """主函数 - 交互式菜单"""
    print("🎯 LangGraph 工具调用系统演示")
    print("="*60)
    
    while True:
        print("\n📋 请选择演示内容:")
        print("1. 🔍 搜索工具演示")
        print("2. 💭 LLM决策演示") 
        print("3. 🔄 状态流转演示")
        print("4. 🚀 完整工作流演示")
        print("5. 📚 查看学习指南")
        print("0. 🚪 退出")
        
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == "1":
            demo_search_tool()
        elif choice == "2":
            demo_llm_decision()
        elif choice == "3":
            demo_state_flow()
        elif choice == "4":
            demo_full_workflow()
        elif choice == "5":
            print("\n📚 学习指南位置: chapter07/graph_learning.md")
            print("📝 测试文件位置: chapter07/76_1_graph_tests.py")
            print("💡 建议按顺序运行演示 1→2→3→4 来理解完整流程")
        elif choice == "0":
            print("\n👋 感谢使用LangGraph演示系统！")
            break
        else:
            print("\n❌ 无效选择，请重新输入")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()
