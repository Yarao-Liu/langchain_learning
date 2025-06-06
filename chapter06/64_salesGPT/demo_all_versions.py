"""
SalesGPT 全版本演示脚本
======================

这个脚本展示了SalesGPT从v1.0到v5.0的所有版本，
让您可以直观地看到每个版本的功能演进。

运行方式：
python demo_all_versions.py

作者：AI助手
日期：2024年
"""

import os
import sys
import time
import warnings
from typing import Dict, Any

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 过滤警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

def print_header(title: str, version: str = ""):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    if version:
        print(f"  {version}")
    print("=" * 80)

def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")

def wait_for_user(message: str = "按回车键继续..."):
    """等待用户输入"""
    input(f"\n{message}")

def demo_version_1():
    """演示v1.0基础版"""
    print_header("SalesGPT v1.0 演示", "基础版 - 最简单的销售对话代理")
    
    try:
        from basic_salesGPT import BasicSalesGPT, llm, demonstrate_basic_sales
        
        print("\n🎯 v1.0 功能特点：")
        print("  ✅ 基础的对话阶段管理")
        print("  ✅ 简单的销售流程")
        print("  ✅ 基本的用户交互")
        print("  ✅ 轻量级实现")
        
        wait_for_user("准备开始v1.0演示，按回车键继续...")
        
        # 运行基础演示
        demonstrate_basic_sales()
        
        print("\n✅ v1.0 演示完成！")
        
    except ImportError as e:
        print(f"❌ 无法导入v1.0模块: {e}")
        print("请确保 01_basic_salesGPT.py 文件存在")
    except Exception as e:
        print(f"❌ v1.0演示出错: {e}")

def demo_version_2():
    """演示v2.0增强对话版"""
    print_header("SalesGPT v2.0 演示", "增强对话版 - 改进的对话管理系统")
    
    try:
        from enhanced_conversation_salesGPT import EnhancedSalesGPT, llm, demonstrate_enhanced_sales
        
        print("\n🎯 v2.0 新增功能：")
        print("  ✅ 智能的阶段分析系统")
        print("  ✅ 更丰富的销售人员信息")
        print("  ✅ 改进的对话管理")
        print("  ✅ 更好的错误处理")
        print("  ✅ 对话状态跟踪")
        
        wait_for_user("准备开始v2.0演示，按回车键继续...")
        
        # 运行增强演示
        demonstrate_enhanced_sales()
        
        print("\n✅ v2.0 演示完成！")
        
    except ImportError as e:
        print(f"❌ 无法导入v2.0模块: {e}")
        print("请确保 02_enhanced_conversation_salesGPT.py 文件存在")
    except Exception as e:
        print(f"❌ v2.0演示出错: {e}")

def demo_version_3():
    """演示v3.0知识库版"""
    print_header("SalesGPT v3.0 演示", "知识库版 - 集成简单知识库系统")
    
    try:
        from knowledge_based_salesGPT import KnowledgeBasedSalesGPT, llm, demonstrate_knowledge_search, demonstrate_knowledge_sales
        
        print("\n🎯 v3.0 新增功能：")
        print("  ✅ 基于关键词的知识库系统")
        print("  ✅ 产品信息管理")
        print("  ✅ 智能信息检索")
        print("  ✅ 上下文感知回复")
        print("  ✅ 结构化产品数据")
        
        wait_for_user("准备开始v3.0演示，按回车键继续...")
        
        # 运行知识库演示
        demonstrate_knowledge_search()
        demonstrate_knowledge_sales()
        
        print("\n✅ v3.0 演示完成！")
        
    except ImportError as e:
        print(f"❌ 无法导入v3.0模块: {e}")
        print("请确保 03_knowledge_based_salesGPT.py 文件存在")
    except Exception as e:
        print(f"❌ v3.0演示出错: {e}")

def demo_version_4():
    """演示v4.0 RAG增强版"""
    print_header("SalesGPT v4.0 演示", "RAG增强版 - 集成向量检索系统")
    
    try:
        from rag_enhanced_salesGPT import RAGEnhancedSalesGPT, llm, demonstrate_rag_knowledge, demonstrate_rag_sales
        
        print("\n🎯 v4.0 新增功能：")
        print("  ✅ 向量嵌入和检索系统")
        print("  ✅ RetrievalQA集成")
        print("  ✅ 智能文档检索")
        print("  ✅ 高级知识问答")
        print("  ✅ 多种嵌入模型支持")
        
        print("\n⚠️  注意：v4.0需要额外依赖：")
        print("  pip install sentence-transformers")
        print("  pip install faiss-cpu")
        
        wait_for_user("准备开始v4.0演示，按回车键继续...")
        
        # 运行RAG演示
        demonstrate_rag_knowledge()
        demonstrate_rag_sales()
        
        print("\n✅ v4.0 演示完成！")
        
    except ImportError as e:
        print(f"❌ 无法导入v4.0模块: {e}")
        print("请确保 04_rag_enhanced_salesGPT.py 文件存在")
        print("并安装必要依赖：pip install sentence-transformers faiss-cpu")
    except Exception as e:
        print(f"❌ v4.0演示出错: {e}")

def demo_version_5():
    """演示v5.0企业版"""
    print_header("SalesGPT v5.0 演示", "企业版 - 完整的企业级销售代理系统")
    
    try:
        from enterprise_salesGPT import EnterpriseSalesGPT, llm, demonstrate_enterprise_features
        
        print("\n🎯 v5.0 新增功能：")
        print("  ✅ 客户档案管理系统")
        print("  ✅ 销售数据分析")
        print("  ✅ 多渠道集成支持")
        print("  ✅ 完整的CRM功能")
        print("  ✅ 销售流程自动化")
        print("  ✅ 性能监控和报告")
        
        wait_for_user("准备开始v5.0演示，按回车键继续...")
        
        # 运行企业版演示
        demonstrate_enterprise_features()
        
        print("\n✅ v5.0 演示完成！")
        
    except ImportError as e:
        print(f"❌ 无法导入v5.0模块: {e}")
        print("请确保 05_enterprise_salesGPT.py 文件存在")
    except Exception as e:
        print(f"❌ v5.0演示出错: {e}")

def show_version_comparison():
    """显示版本对比"""
    print_header("SalesGPT 版本功能对比")
    
    comparison_table = """
| 功能特性           | v1.0 | v2.0 | v3.0 | v4.0 | v5.0 |
|-------------------|------|------|------|------|------|
| 基础对话          | ✅   | ✅   | ✅   | ✅   | ✅   |
| 智能阶段分析      | ❌   | ✅   | ✅   | ✅   | ✅   |
| 关键词知识库      | ❌   | ❌   | ✅   | ❌   | ❌   |
| 向量检索          | ❌   | ❌   | ❌   | ✅   | ✅   |
| 客户管理          | ❌   | ❌   | ❌   | ❌   | ✅   |
| 数据分析          | ❌   | ❌   | ❌   | ❌   | ✅   |
| 多渠道支持        | ❌   | ❌   | ❌   | ❌   | ✅   |
| 复杂度            | 低   | 中   | 中   | 高   | 很高 |
| 适用场景          | 学习 | 原型 | 演示 | 产品 | 企业 |
    """
    
    print(comparison_table)

def show_learning_path():
    """显示学习路径建议"""
    print_header("学习路径建议")
    
    print("\n🎓 初学者路径：")
    print("  1. v1.0 基础版 - 理解基本概念")
    print("  2. v2.0 增强版 - 学习提示词工程")
    print("  3. v3.0 知识库版 - 了解信息检索")
    
    print("\n🚀 进阶路径：")
    print("  1. v4.0 RAG增强版 - 掌握向量检索")
    print("  2. v5.0 企业版 - 学习系统架构")
    
    print("\n💼 实际应用路径：")
    print("  • 快速原型: 使用 v1.0 或 v2.0")
    print("  • 产品演示: 使用 v3.0 或 v4.0")
    print("  • 生产部署: 使用 v5.0")

def main():
    """主函数"""
    print_header("SalesGPT 全版本演示系统", "从简单到复杂的渐进式开发演示")
    
    print("\n🎯 本演示将展示SalesGPT从v1.0到v5.0的完整演进过程")
    print("每个版本都在前一版本基础上增加新功能，展示了AI销售代理的发展历程")
    
    while True:
        print("\n" + "=" * 60)
        print("请选择要演示的版本：")
        print("=" * 60)
        print("1. v1.0 - 基础版")
        print("2. v2.0 - 增强对话版")
        print("3. v3.0 - 知识库版")
        print("4. v4.0 - RAG增强版")
        print("5. v5.0 - 企业版")
        print("6. 版本功能对比")
        print("7. 学习路径建议")
        print("8. 全部演示（按顺序）")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-8): ").strip()
        
        if choice == "1":
            demo_version_1()
        elif choice == "2":
            demo_version_2()
        elif choice == "3":
            demo_version_3()
        elif choice == "4":
            demo_version_4()
        elif choice == "5":
            demo_version_5()
        elif choice == "6":
            show_version_comparison()
        elif choice == "7":
            show_learning_path()
        elif choice == "8":
            print("\n🚀 开始全版本演示...")
            demo_version_1()
            demo_version_2()
            demo_version_3()
            demo_version_4()
            demo_version_5()
            show_version_comparison()
            print("\n🎉 全版本演示完成！")
        elif choice == "0":
            print("\n👋 感谢使用SalesGPT演示系统！")
            break
        else:
            print("\n❌ 无效选择，请重新输入")
        
        if choice != "0":
            wait_for_user("\n按回车键返回主菜单...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请检查所有文件是否存在并且依赖已正确安装")
