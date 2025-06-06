"""
RAG增强版 SalesGPT v4.0 - 集成向量检索系统
==========================================

这是SalesGPT系列的第四个版本，在知识库版基础上增加了：
1. 向量嵌入和检索系统
2. RetrievalQA集成
3. 智能文档检索
4. 高级知识问答
5. 多种嵌入模型支持

功能特点：
- 基于向量的语义检索
- RetrievalQA问答系统
- 多种嵌入模型备选
- 智能文档分割
- 上下文相关的知识检索

作者：AI助手
日期：2024年
版本：4.0 - RAG增强版
"""

import os
import warnings
import json
from typing import Dict, Any, List

import dotenv
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# 尝试导入不同的嵌入模型
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False

# 过滤弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 加载环境变量
dotenv.load_dotenv()

# 检查 API 密钥
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("错误: 未找到 OPENAI_API_KEY 环境变量")
    print("请在 .env 文件中设置: OPENAI_API_KEY=your_api_key_here")
    exit(1)

# 创建 LLM
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1/",
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.1
)

# 销售对话阶段定义
SALES_STAGES = {
    "1": "介绍：通过介绍自己和公司来开始对话，建立信任关系",
    "2": "资格确认：确认客户是否是合适的潜在客户，了解决策权限",
    "3": "需求分析：深入了解客户的具体需求、痛点和期望",
    "4": "价值主张：展示产品/服务的独特价值和竞争优势",
    "5": "解决方案展示：根据客户需求展示具体的解决方案",
    "6": "异议处理：处理客户的疑虑、反对意见和担忧",
    "7": "成交推进：推进销售进程，提出具体的下一步行动"
}

class RAGKnowledgeBase:
    """基于RAG的知识库系统"""
    
    def __init__(self, knowledge_file_path: str = None):
        """初始化RAG知识库"""
        self.knowledge_file_path = knowledge_file_path or "chapter06/data/car_knowledge_base.txt"
        self.vectorstore = None
        self.qa_chain = None
        self.setup_knowledge_base()
    
    def setup_knowledge_base(self):
        """设置知识库和向量检索"""
        try:
            print("正在设置RAG知识库...")
            
            # 检查知识库文件是否存在
            if not os.path.exists(self.knowledge_file_path):
                print(f"知识库文件不存在: {self.knowledge_file_path}")
                self._create_default_knowledge_file()
            
            # 加载文档
            loader = TextLoader(self.knowledge_file_path, encoding='utf-8')
            documents = loader.load()
            
            # 分割文档
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n"
            )
            texts = text_splitter.split_documents(documents)
            
            # 创建嵌入模型
            embeddings = self._get_embeddings()
            if not embeddings:
                print("无法创建嵌入模型，RAG功能将被禁用")
                return
            
            # 创建向量存储
            self.vectorstore = FAISS.from_documents(texts, embeddings)
            
            # 创建检索问答链
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            print("✅ RAG知识库设置完成！")
            
        except Exception as e:
            print(f"❌ RAG知识库设置失败: {e}")
            self.vectorstore = None
            self.qa_chain = None
    
    def _get_embeddings(self):
        """获取嵌入模型"""
        embeddings = None
        
        # 尝试使用HuggingFace嵌入模型
        if HUGGINGFACE_AVAILABLE:
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                print("使用 HuggingFace 嵌入模型")
                return embeddings
            except Exception as e:
                print(f"HuggingFace 嵌入模型不可用: {e}")
        
        # 尝试使用OpenAI嵌入模型
        if OPENAI_EMBEDDINGS_AVAILABLE:
            try:
                embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    openai_api_base="https://api.siliconflow.cn/v1/",
                    model="text-embedding-ada-002"
                )
                print("使用 OpenAI 嵌入模型")
                return embeddings
            except Exception as e:
                print(f"OpenAI 嵌入模型不可用: {e}")
        
        return None
    
    def _create_default_knowledge_file(self):
        """创建默认知识库文件"""
        default_knowledge = """
智能科技产品知识库
==================

## 智能办公系统

### 产品概述
智能办公系统是一款基于AI技术的企业办公解决方案，旨在提升企业工作效率和协作能力。

### 核心功能
- 智能文档管理：自动分类、标签和检索
- 自动化工作流：流程自动化和任务分配
- AI助手：智能问答和决策支持
- 数据分析：业务数据可视化和洞察
- 团队协作：实时协作和沟通工具

### 技术优势
- 采用最新的自然语言处理技术
- 支持多种文档格式和数据源
- 云端部署，安全可靠
- 移动端支持，随时随地办公

### 价格方案
- 基础版：8万元/年，支持50用户
- 专业版：15万元/年，支持200用户
- 企业版：25万元/年，支持1000用户

## AI数据分析平台

### 产品概述
AI数据分析平台是一款企业级的数据分析和商业智能解决方案。

### 核心功能
- 实时数据处理和分析
- 智能报表生成
- 预测性分析
- 可视化展示
- API接口集成

### 应用场景
- 销售数据分析
- 客户行为分析
- 运营效率分析
- 风险预测和控制

### 价格方案
- 标准版：12万元/年
- 高级版：20万元/年
- 定制版：根据需求报价

## 智能客服机器人

### 产品概述
智能客服机器人是基于AI技术的客户服务解决方案。

### 核心功能
- 24小时在线服务
- 多渠道接入支持
- 智能问答和对话
- 情感分析
- 人工客服无缝转接

### 技术特点
- 自然语言理解
- 机器学习优化
- 多语言支持
- 个性化服务

### 价格方案
- 基础版：5万元/年
- 增强版：10万元/年
- 定制版：15万元/年

## 服务支持

### 技术支持
- 7×24小时技术支持
- 专业工程师团队
- 远程协助和现场服务
- 定期系统维护和升级

### 培训服务
- 用户操作培训
- 管理员培训
- 高级功能培训
- 在线培训资源

### 实施服务
- 需求分析和方案设计
- 系统部署和配置
- 数据迁移和集成
- 用户验收和上线

## 公司介绍

### 公司概况
智能科技有限公司成立于2015年，专注于企业AI解决方案的研发和服务。

### 技术实力
- 拥有50+技术专家
- 10年AI技术积累
- 多项核心技术专利
- 与知名高校合作

### 客户案例
- 服务500+企业客户
- 包括多家世界500强企业
- 客户满意度98%
- 续约率95%

### 行业经验
- 制造业数字化转型
- 金融行业智能化升级
- 教育行业信息化建设
- 电商行业数据分析
        """
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.knowledge_file_path), exist_ok=True)
        
        # 写入默认知识库
        with open(self.knowledge_file_path, 'w', encoding='utf-8') as f:
            f.write(default_knowledge)
        
        print(f"✅ 已创建默认知识库文件: {self.knowledge_file_path}")
    
    def query(self, question: str) -> str:
        """查询知识库"""
        if not self.qa_chain:
            return "抱歉，知识库暂时不可用。"

        try:
            result = self.qa_chain.invoke({"query": question})
            return result["result"]
        except Exception as e:
            print(f"知识库查询错误: {e}")
            return "抱歉，查询过程中出现了问题。"

class StageAnalyzer:
    """智能阶段分析器"""

    def __init__(self, llm):
        self.llm = llm
        self.analyzer_chain = self._create_analyzer_chain()

    def _create_analyzer_chain(self):
        """创建阶段分析链"""
        prompt_template = """
你是销售对话阶段分析专家。根据对话历史，判断销售对话应该进入哪个阶段。

对话历史：
{conversation_history}

销售阶段选项：
1. 介绍：通过介绍自己和公司来开始对话，建立信任关系
2. 资格确认：确认客户是否是合适的潜在客户，了解决策权限
3. 需求分析：深入了解客户的具体需求、痛点和期望
4. 价值主张：展示产品/服务的独特价值和竞争优势
5. 解决方案展示：根据客户需求展示具体的解决方案
6. 异议处理：处理客户的疑虑、反对意见和担忧
7. 成交推进：推进销售进程，提出具体的下一步行动

只回答数字1-7：
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["conversation_history"]
        )

        return LLMChain(prompt=prompt, llm=self.llm, verbose=False)

    def analyze_stage(self, conversation_history: str) -> str:
        """分析当前应该进入的阶段"""
        try:
            result = self.analyzer_chain.invoke({"conversation_history": conversation_history})
            stage = result.get("text", "1").strip()
            return stage if stage in SALES_STAGES else "1"
        except Exception as e:
            print(f"阶段分析错误: {e}")
            return "1"

class RAGEnhancedSalesGPT:
    """RAG增强版销售对话代理"""

    def __init__(self, llm, knowledge_file_path: str = None, verbose=True):
        """初始化销售代理"""
        self.llm = llm
        self.verbose = verbose
        self.knowledge_base = RAGKnowledgeBase(knowledge_file_path)
        self.stage_analyzer = StageAnalyzer(llm)
        self.conversation_history = []
        self.current_stage = "1"

        # 销售人员信息
        self.salesperson_info = {
            "name": "小张",
            "role": "高级解决方案专家",
            "company": "智能科技有限公司",
            "company_business": "专注于为企业提供AI驱动的智能化解决方案，包括智能办公、数据分析、客服机器人等产品",
            "company_values": "以技术创新为驱动，为客户创造价值，推动企业数字化转型",
            "contact_purpose": "了解客户的数字化需求，基于我们的产品知识库为客户提供最专业的解决方案"
        }

        # 创建对话链
        self.conversation_chain = self._create_conversation_chain()

    def _create_conversation_chain(self):
        """创建RAG增强的对话链"""
        prompt_template = """
你是{name}，{role}，在{company}工作。

公司业务：{company_business}
公司价值观：{company_values}
联系目的：{contact_purpose}

当前销售阶段：{current_stage}
阶段说明：{stage_description}

基于知识库的相关信息：
{knowledge_context}

对话历史：
{conversation_history}

请根据当前阶段和知识库信息生成专业回复：
- 充分利用知识库提供的准确信息
- 根据客户问题提供详细和专业的回答
- 保持专业、友好和有帮助的语调
- 适时推进销售进程
- 如果知识库中没有相关信息，诚实说明并提供一般性建议
- 以 <END_OF_TURN> 结尾

{name}：
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "name", "role", "company", "company_business", "company_values",
                "contact_purpose", "current_stage", "stage_description",
                "knowledge_context", "conversation_history"
            ]
        )

        return LLMChain(prompt=prompt, llm=self.llm, verbose=self.verbose)

    def get_knowledge_context(self, user_input: str) -> str:
        """获取相关的产品知识"""
        if not user_input:
            return ""

        # 检查用户输入是否包含产品相关关键词
        product_keywords = ["产品", "价格", "功能", "服务", "技术", "解决方案", "系统", "平台", "机器人"]

        if any(keyword in user_input for keyword in product_keywords):
            knowledge = self.knowledge_base.query(user_input)
            return f"相关产品信息：{knowledge}"

        return ""

    def step(self, user_input: str = None) -> str:
        """执行一步对话"""
        # 获取知识上下文
        knowledge_context = ""
        if user_input:
            knowledge_context = self.get_knowledge_context(user_input)
            self.conversation_history.append(f"客户：{user_input}<END_OF_TURN>")

        # 构建对话历史
        history_str = "".join(self.conversation_history[-10:])
        if not history_str:
            history_str = "对话开始"

        # 分析当前阶段
        if len(self.conversation_history) > 0:
            self.current_stage = self.stage_analyzer.analyze_stage(history_str)

        # 生成回复
        try:
            result = self.conversation_chain.invoke({
                **self.salesperson_info,
                "current_stage": self.current_stage,
                "stage_description": SALES_STAGES[self.current_stage],
                "knowledge_context": knowledge_context,
                "conversation_history": history_str
            })

            response = result.get("text", "").strip()
            self.conversation_history.append(f"{self.salesperson_info['name']}：{response}<END_OF_TURN>")

            return response

        except Exception as e:
            print(f"生成回复时出错: {e}")
            return "抱歉，我遇到了技术问题，请稍后再试。"

    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        return {
            "current_stage": self.current_stage,
            "stage_description": SALES_STAGES[self.current_stage],
            "conversation_turns": len(self.conversation_history),
            "salesperson": self.salesperson_info["name"],
            "company": self.salesperson_info["company"],
            "rag_enabled": self.knowledge_base.qa_chain is not None
        }

def demonstrate_rag_knowledge():
    """演示RAG知识库功能"""
    print("=" * 60)
    print("RAG知识库功能演示")
    print("=" * 60)

    # 初始化RAG知识库
    kb = RAGKnowledgeBase()

    if not kb.qa_chain:
        print("RAG知识库初始化失败，跳过演示")
        return

    # 测试问题
    test_questions = [
        "智能办公系统有什么功能？",
        "AI数据分析平台的价格是多少？",
        "你们提供什么技术支持？",
        "智能客服机器人的优势是什么？",
        "公司有哪些客户案例？"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. 问题: {question}")
        answer = kb.query(question)
        print(f"   回答: {answer}")

def demonstrate_rag_sales():
    """演示RAG增强版销售对话"""
    print("\n" + "=" * 60)
    print("RAG增强版销售对话演示")
    print("=" * 60)

    # 创建RAG增强版销售代理
    sales_agent = RAGEnhancedSalesGPT(llm, verbose=False)

    print(f"\n{sales_agent.salesperson_info['name']}开始对话...")
    response = sales_agent.step()
    print(f"\n{sales_agent.salesperson_info['name']}: {response}")

    # 模拟客户对话
    customer_inputs = [
        "你好，我想了解你们的产品",
        "我们公司需要一个办公系统，你们有什么推荐？",
        "智能办公系统的价格怎么样？",
        "具体有哪些功能？实施起来复杂吗？",
        "你们的技术支持怎么样？",
        "有没有类似的客户案例可以参考？"
    ]

    for customer_input in customer_inputs:
        print(f"\n客户: {customer_input}")
        response = sales_agent.step(customer_input)
        print(f"{sales_agent.salesperson_info['name']}: {response}")

        summary = sales_agent.get_conversation_summary()
        print(f"[阶段 {summary['current_stage']}: {summary['stage_description']} | RAG: {'✅' if summary['rag_enabled'] else '❌'}]")

def interactive_rag_demo():
    """交互式RAG演示"""
    print("\n" + "=" * 60)
    print("交互式RAG增强销售对话")
    print("=" * 60)
    print("输入 'quit' 退出，'summary' 查看对话摘要，'knowledge <问题>' 直接查询知识库")
    print("-" * 60)

    sales_agent = RAGEnhancedSalesGPT(llm, verbose=False)

    print(f"\n{sales_agent.salesperson_info['name']}开始对话...")
    response = sales_agent.step()
    print(f"\n{sales_agent.salesperson_info['name']}: {response}")

    while True:
        try:
            user_input = input(f"\n您: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n感谢您的咨询，期待为您服务！")
                break

            if user_input.lower() == 'summary':
                summary = sales_agent.get_conversation_summary()
                print(f"\n对话摘要:")
                print(f"- 当前阶段: {summary['current_stage']} - {summary['stage_description']}")
                print(f"- 对话轮数: {summary['conversation_turns']}")
                print(f"- 销售专家: {summary['salesperson']} ({summary['company']})")
                print(f"- RAG状态: {'✅ 已启用' if summary['rag_enabled'] else '❌ 未启用'}")
                continue

            if user_input.lower().startswith('knowledge '):
                question = user_input[10:]  # 去掉 'knowledge ' 前缀
                if question:
                    answer = sales_agent.knowledge_base.query(question)
                    print(f"\n知识库回答: {answer}")
                else:
                    print("\n请在 'knowledge' 后面输入问题")
                continue

            if not user_input:
                continue

            response = sales_agent.step(user_input)
            print(f"\n{sales_agent.salesperson_info['name']}: {response}")

        except KeyboardInterrupt:
            print("\n\n对话结束")
            break
        except Exception as e:
            print(f"\n出现错误: {e}")

def main():
    """主函数"""
    print("RAG增强版 SalesGPT v4.0")
    print("集成向量检索系统的智能销售代理")

    try:
        # 1. 演示RAG知识库功能
        demonstrate_rag_knowledge()

        # 2. 演示RAG增强版销售对话
        demonstrate_rag_sales()

        # 3. 交互式演示
        choice = input("\n是否要进行交互式对话？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_rag_demo()

    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"\n\n程序出错: {e}")

    finally:
        print("\n" + "=" * 60)
        print("RAG增强版 SalesGPT v4.0 演示结束")
        print("=" * 60)

if __name__ == "__main__":
    main()
