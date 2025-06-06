"""
SalesGPT 导入测试脚本
===================

这个脚本测试所有版本的SalesGPT是否能正常导入和初始化。

运行方式：
python test_imports.py
"""

import os
import sys
import warnings

# 过滤警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_version_1():
    """测试v1.0基础版"""
    try:
        # 修改导入路径
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # 重命名导入以避免冲突
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("basic_salesGPT", "01_basic_salesGPT.py")
        basic_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(basic_module)
        
        # 测试类是否可以实例化
        sales_agent = basic_module.BasicSalesGPT(basic_module.llm)
        print("✅ v1.0 基础版 - 导入成功")
        return True
    except Exception as e:
        print(f"❌ v1.0 基础版 - 导入失败: {e}")
        return False

def test_version_2():
    """测试v2.0增强对话版"""
    try:
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("enhanced_conversation_salesGPT",
                                                      "02_enhanced_conversation_salesGPT.py")
        enhanced_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enhanced_module)
        
        sales_agent = enhanced_module.EnhancedSalesGPT(enhanced_module.llm, verbose=False)
        print("✅ v2.0 增强对话版 - 导入成功")
        return True
    except Exception as e:
        print(f"❌ v2.0 增强对话版 - 导入失败: {e}")
        return False

def test_version_3():
    """测试v3.0知识库版"""
    try:
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("knowledge_based_salesGPT", "03_knowledge_based_salesGPT.py")
        knowledge_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(knowledge_module)
        
        sales_agent = knowledge_module.KnowledgeBasedSalesGPT(knowledge_module.llm, verbose=False)
        print("✅ v3.0 知识库版 - 导入成功")
        return True
    except Exception as e:
        print(f"❌ v3.0 知识库版 - 导入失败: {e}")
        return False

def test_version_4():
    """测试v4.0 RAG增强版"""
    try:
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("rag_enhanced_salesGPT", "04_rag_enhanced_salesGPT.py")
        rag_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_module)
        
        sales_agent = rag_module.RAGEnhancedSalesGPT(rag_module.llm, verbose=False)
        print("✅ v4.0 RAG增强版 - 导入成功")
        return True
    except Exception as e:
        print(f"❌ v4.0 RAG增强版 - 导入失败: {e}")
        print("  提示：可能需要安装额外依赖：pip install sentence-transformers faiss-cpu")
        return False

def test_version_5():
    """测试v5.0企业版"""
    try:
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("enterprise_salesGPT", "05_enterprise_salesGPT.py")
        enterprise_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enterprise_module)
        
        sales_agent = enterprise_module.EnterpriseSalesGPT(enterprise_module.llm, verbose=False)
        print("✅ v5.0 企业版 - 导入成功")
        return True
    except Exception as e:
        print(f"❌ v5.0 企业版 - 导入失败: {e}")
        return False

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        "langchain",
        "langchain_openai", 
        "langchain_community",
        "python-dotenv"
    ]
    
    optional_packages = [
        "sentence-transformers",
        "faiss-cpu"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - 必需依赖，请安装")
    
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ⚠️  {package} - 可选依赖，v4.0和v5.0需要")

def check_environment():
    """检查环境配置"""
    print("\n🔍 检查环境配置...")
    
    # 检查.env文件
    if os.path.exists(".env"):
        print("  ✅ .env 文件存在")
    else:
        print("  ⚠️  .env 文件不存在，请创建并设置OPENAI_API_KEY")
    
    # 检查API密钥
    import dotenv
    dotenv.load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("  ✅ OPENAI_API_KEY 已设置")
    else:
        print("  ❌ OPENAI_API_KEY 未设置")

def main():
    """主函数"""
    print("=" * 60)
    print("SalesGPT 导入测试")
    print("=" * 60)
    
    # 检查依赖和环境
    check_dependencies()
    check_environment()
    
    print("\n🧪 测试各版本导入...")
    print("-" * 40)
    
    # 测试各版本
    results = []
    results.append(("v1.0 基础版", test_version_1()))
    results.append(("v2.0 增强对话版", test_version_2()))
    results.append(("v3.0 知识库版", test_version_3()))
    results.append(("v4.0 RAG增强版", test_version_4()))
    results.append(("v5.0 企业版", test_version_5()))
    
    # 汇总结果
    print("\n📊 测试结果汇总:")
    print("-" * 40)
    
    success_count = 0
    for version, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {version}: {status}")
        if success:
            success_count += 1
    
    print(f"\n🎯 总体结果: {success_count}/{len(results)} 个版本测试通过")
    
    if success_count == len(results):
        print("🎉 所有版本都可以正常使用！")
    elif success_count >= 3:
        print("👍 大部分版本可以正常使用")
    else:
        print("⚠️  请检查依赖安装和环境配置")
    
    print("\n💡 使用建议:")
    if success_count >= 1:
        print("  • 可以从成功的版本开始学习")
    if success_count < len(results):
        print("  • 安装缺失的依赖包")
        print("  • 检查.env文件配置")
    
    print("\n📚 更多信息请查看 README.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
