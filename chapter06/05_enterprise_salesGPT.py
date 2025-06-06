"""
企业版 SalesGPT v5.0 - 完整的企业级销售代理系统
===============================================

这是SalesGPT系列的最终版本，在RAG增强版基础上增加了：
1. 客户档案管理系统
2. 销售数据分析
3. 多渠道集成支持
4. 完整的CRM功能
5. 销售流程自动化
6. 性能监控和报告

功能特点：
- 完整的客户生命周期管理
- 智能销售预测和分析
- 多渠道统一管理
- 自动化销售流程
- 实时性能监控
- 详细的销售报告

作者：AI助手
日期：2024年
版本：5.0 - 企业版
"""

import os
import warnings
import json
import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import dotenv
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# 尝试导入嵌入模型
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

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

class CustomerStatus(Enum):
    """客户状态枚举"""
    LEAD = "潜在客户"
    QUALIFIED = "合格客户"
    OPPORTUNITY = "销售机会"
    PROPOSAL = "方案阶段"
    NEGOTIATION = "谈判阶段"
    CLOSED_WON = "成交"
    CLOSED_LOST = "流失"

class InteractionChannel(Enum):
    """交互渠道枚举"""
    PHONE = "电话"
    EMAIL = "邮件"
    CHAT = "在线聊天"
    MEETING = "面对面会议"
    VIDEO_CALL = "视频通话"

@dataclass
class CustomerProfile:
    """客户档案数据类"""
    customer_id: str
    name: str
    company: str
    position: str
    email: str
    phone: str
    industry: str
    company_size: str
    budget_range: str
    pain_points: List[str]
    interests: List[str]
    status: CustomerStatus
    created_at: str
    last_contact: str
    notes: str = ""

@dataclass
class SalesInteraction:
    """销售互动记录数据类"""
    interaction_id: str
    customer_id: str
    timestamp: str
    channel: InteractionChannel
    stage: str
    content: str
    outcome: str
    next_action: str
    salesperson: str

class CustomerManager:
    """客户管理系统"""
    
    def __init__(self):
        """初始化客户管理器"""
        self.customers: Dict[str, CustomerProfile] = {}
        self.interactions: List[SalesInteraction] = []
        self.load_customer_data()
    
    def load_customer_data(self):
        """加载客户数据"""
        try:
            # 尝试从文件加载客户数据
            if os.path.exists("data/customers.json"):
                with open("data/customers.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for customer_data in data.get("customers", []):
                        customer = CustomerProfile(**customer_data)
                        self.customers[customer.customer_id] = customer
            
            # 加载交互记录
            if os.path.exists("data/interactions.json"):
                with open("data/interactions.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for interaction_data in data.get("interactions", []):
                        interaction = SalesInteraction(**interaction_data)
                        self.interactions.append(interaction)
        
        except Exception as e:
            print(f"加载客户数据失败: {e}")
            self._create_sample_customers()
    
    def _create_sample_customers(self):
        """创建示例客户数据"""
        sample_customers = [
            CustomerProfile(
                customer_id="CUST001",
                name="张总",
                company="创新科技有限公司",
                position="CEO",
                email="zhang@innovation.com",
                phone="13800138001",
                industry="软件开发",
                company_size="100-500人",
                budget_range="20-50万",
                pain_points=["办公效率低", "数据分析能力不足", "客服成本高"],
                interests=["AI技术", "自动化", "数据分析"],
                status=CustomerStatus.QUALIFIED,
                created_at=datetime.datetime.now().isoformat(),
                last_contact=datetime.datetime.now().isoformat(),
                notes="对AI解决方案很感兴趣，决策权限高"
            ),
            CustomerProfile(
                customer_id="CUST002",
                name="李经理",
                company="制造业集团",
                position="IT经理",
                email="li@manufacturing.com",
                phone="13800138002",
                industry="制造业",
                company_size="500-1000人",
                budget_range="50-100万",
                pain_points=["生产效率", "质量控制", "成本控制"],
                interests=["智能制造", "数据分析", "自动化"],
                status=CustomerStatus.OPPORTUNITY,
                created_at=datetime.datetime.now().isoformat(),
                last_contact=datetime.datetime.now().isoformat(),
                notes="正在评估多个供应商，价格敏感"
            )
        ]
        
        for customer in sample_customers:
            self.customers[customer.customer_id] = customer
    
    def get_customer(self, customer_id: str) -> Optional[CustomerProfile]:
        """获取客户信息"""
        return self.customers.get(customer_id)
    
    def update_customer_status(self, customer_id: str, status: CustomerStatus):
        """更新客户状态"""
        if customer_id in self.customers:
            self.customers[customer_id].status = status
            self.customers[customer_id].last_contact = datetime.datetime.now().isoformat()
    
    def add_interaction(self, interaction: SalesInteraction):
        """添加交互记录"""
        self.interactions.append(interaction)
        # 更新客户最后联系时间
        if interaction.customer_id in self.customers:
            self.customers[interaction.customer_id].last_contact = interaction.timestamp
    
    def get_customer_interactions(self, customer_id: str) -> List[SalesInteraction]:
        """获取客户的所有交互记录"""
        return [i for i in self.interactions if i.customer_id == customer_id]
    
    def save_data(self):
        """保存客户数据"""
        try:
            os.makedirs("data", exist_ok=True)
            
            # 保存客户数据
            customers_data = {
                "customers": [asdict(customer) for customer in self.customers.values()]
            }
            with open("data/customers.json", "w", encoding="utf-8") as f:
                json.dump(customers_data, f, ensure_ascii=False, indent=2)
            
            # 保存交互记录
            interactions_data = {
                "interactions": [asdict(interaction) for interaction in self.interactions]
            }
            with open("data/interactions.json", "w", encoding="utf-8") as f:
                json.dump(interactions_data, f, ensure_ascii=False, indent=2)
            
            print("✅ 客户数据已保存")
        
        except Exception as e:
            print(f"❌ 保存客户数据失败: {e}")

class SalesAnalytics:
    """销售数据分析系统"""
    
    def __init__(self, customer_manager: CustomerManager):
        """初始化分析系统"""
        self.customer_manager = customer_manager
    
    def get_sales_summary(self) -> Dict[str, Any]:
        """获取销售摘要"""
        customers = list(self.customer_manager.customers.values())
        interactions = self.customer_manager.interactions
        
        # 统计客户状态分布
        status_count = {}
        for customer in customers:
            status = customer.status.value
            status_count[status] = status_count.get(status, 0) + 1
        
        # 统计交互渠道分布
        channel_count = {}
        for interaction in interactions:
            channel = interaction.channel.value
            channel_count[channel] = channel_count.get(channel, 0) + 1
        
        # 统计阶段分布
        stage_count = {}
        for interaction in interactions:
            stage = interaction.stage
            stage_count[stage] = stage_count.get(stage, 0) + 1
        
        return {
            "total_customers": len(customers),
            "total_interactions": len(interactions),
            "status_distribution": status_count,
            "channel_distribution": channel_count,
            "stage_distribution": stage_count,
            "conversion_rate": self._calculate_conversion_rate(customers)
        }
    
    def _calculate_conversion_rate(self, customers: List[CustomerProfile]) -> float:
        """计算转化率"""
        if not customers:
            return 0.0
        
        closed_won = sum(1 for c in customers if c.status == CustomerStatus.CLOSED_WON)
        return (closed_won / len(customers)) * 100
    
    def get_customer_insights(self, customer_id: str) -> Dict[str, Any]:
        """获取客户洞察"""
        customer = self.customer_manager.get_customer(customer_id)
        if not customer:
            return {}
        
        interactions = self.customer_manager.get_customer_interactions(customer_id)
        
        return {
            "customer_profile": asdict(customer),
            "interaction_count": len(interactions),
            "last_interaction": interactions[-1].timestamp if interactions else None,
            "engagement_score": self._calculate_engagement_score(interactions),
            "recommended_actions": self._get_recommended_actions(customer, interactions)
        }
    
    def _calculate_engagement_score(self, interactions: List[SalesInteraction]) -> int:
        """计算客户参与度评分"""
        if not interactions:
            return 0
        
        # 简单的评分算法：基于交互频率和最近活跃度
        recent_interactions = len([i for i in interactions[-5:]])  # 最近5次交互
        total_interactions = len(interactions)
        
        score = min(100, (recent_interactions * 20) + (total_interactions * 5))
        return score
    
    def _get_recommended_actions(self, customer: CustomerProfile, interactions: List[SalesInteraction]) -> List[str]:
        """获取推荐行动"""
        actions = []
        
        if not interactions:
            actions.append("建立初次联系")
        elif len(interactions) < 3:
            actions.append("增加互动频率")
        
        if customer.status == CustomerStatus.LEAD:
            actions.append("进行资格确认")
        elif customer.status == CustomerStatus.QUALIFIED:
            actions.append("深入需求分析")
        elif customer.status == CustomerStatus.OPPORTUNITY:
            actions.append("准备解决方案提案")
        
        return actions

class EnterpriseKnowledgeBase:
    """企业级知识库系统"""

    def __init__(self, knowledge_file_path: str = None):
        """初始化企业知识库"""
        self.knowledge_file_path = knowledge_file_path or "chapter07/data/car_knowledge_base.txt"
        self.vectorstore = None
        self.qa_chain = None
        self.setup_knowledge_base()

    def setup_knowledge_base(self):
        """设置知识库"""
        try:
            print("正在设置企业级知识库...")

            if not os.path.exists(self.knowledge_file_path):
                self._create_enterprise_knowledge_file()

            loader = TextLoader(self.knowledge_file_path, encoding='utf-8')
            documents = loader.load()

            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n"
            )
            texts = text_splitter.split_documents(documents)

            if HUGGINGFACE_AVAILABLE:
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )
                    self.vectorstore = FAISS.from_documents(texts, embeddings)

                    self.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )
                    print("✅ 企业级知识库设置完成！")
                except Exception as e:
                    print(f"❌ 知识库设置失败: {e}")
            else:
                print("❌ 缺少必要的依赖，知识库功能被禁用")

        except Exception as e:
            print(f"❌ 企业级知识库设置失败: {e}")

    def _create_enterprise_knowledge_file(self):
        """创建企业级知识库文件"""
        enterprise_knowledge = """
企业级AI解决方案知识库
====================

## 产品组合

### 智能办公套件
- 文档智能管理系统
- 自动化工作流引擎
- AI会议助手
- 智能日程管理
- 团队协作平台

### 数据智能平台
- 实时数据处理引擎
- 商业智能分析
- 预测性分析模型
- 数据可视化工具
- 自助式分析平台

### 客户服务解决方案
- 智能客服机器人
- 多渠道客户服务
- 客户情感分析
- 服务质量监控
- 知识库管理

## 行业解决方案

### 制造业
- 智能制造执行系统
- 质量管理系统
- 供应链优化
- 设备预测性维护
- 生产计划优化

### 金融服务
- 风险管理系统
- 反欺诈检测
- 智能投顾
- 客户画像分析
- 合规监控

### 零售电商
- 个性化推荐引擎
- 库存优化管理
- 价格策略优化
- 客户行为分析
- 供应链管理

## 技术架构

### 云原生架构
- 微服务架构设计
- 容器化部署
- 自动扩缩容
- 高可用性保障
- 灾备恢复

### 安全保障
- 数据加密传输
- 身份认证授权
- 审计日志记录
- 合规性保障
- 隐私保护

## 服务体系

### 咨询服务
- 数字化转型咨询
- 技术架构设计
- 业务流程优化
- 变更管理支持
- ROI评估分析

### 实施服务
- 项目管理
- 系统集成
- 数据迁移
- 用户培训
- 上线支持

### 运维服务
- 7×24小时监控
- 性能优化
- 安全维护
- 版本升级
- 技术支持

## 成功案例

### 大型制造企业
- 项目规模：1000+用户
- 实施周期：6个月
- 效果：生产效率提升30%
- ROI：18个月回本

### 金融机构
- 项目规模：5000+用户
- 实施周期：12个月
- 效果：风险识别准确率95%
- ROI：24个月回本

### 零售连锁
- 项目规模：500+门店
- 实施周期：9个月
- 效果：销售额增长25%
- ROI：15个月回本
        """

        os.makedirs(os.path.dirname(self.knowledge_file_path), exist_ok=True)
        with open(self.knowledge_file_path, 'w', encoding='utf-8') as f:
            f.write(enterprise_knowledge)

        print(f"✅ 已创建企业级知识库文件: {self.knowledge_file_path}")

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

class EnterpriseSalesGPT:
    """企业级销售代理系统"""

    def __init__(self, llm, customer_id: str = None, verbose=True):
        """初始化企业级销售代理"""
        self.llm = llm
        self.verbose = verbose
        self.customer_id = customer_id

        # 初始化各个组件
        self.customer_manager = CustomerManager()
        self.analytics = SalesAnalytics(self.customer_manager)
        self.knowledge_base = EnterpriseKnowledgeBase()

        # 对话状态
        self.conversation_history = []
        self.current_stage = "1"
        self.current_channel = InteractionChannel.CHAT

        # 销售人员信息
        self.salesperson_info = {
            "name": "小王",
            "role": "企业级解决方案专家",
            "company": "智能科技有限公司",
            "company_business": "为企业提供全方位的AI驱动数字化转型解决方案",
            "experience": "8年企业级项目经验",
            "specialization": "制造业、金融、零售行业数字化转型"
        }

        # 创建对话链
        self.conversation_chain = self._create_conversation_chain()
        self.stage_analyzer_chain = self._create_stage_analyzer_chain()

    def _create_conversation_chain(self):
        """创建企业级对话链"""
        prompt_template = """
你是{name}，{role}，在{company}工作，拥有{experience}。
专业领域：{specialization}

公司业务：{company_business}

当前客户信息：
{customer_context}

当前销售阶段：{current_stage}
阶段说明：{stage_description}

相关产品知识：
{knowledge_context}

对话历史：
{conversation_history}

请根据客户信息、当前阶段和产品知识生成专业回复：
- 充分利用客户档案信息进行个性化沟通
- 基于客户的行业和需求提供针对性建议
- 利用知识库提供准确的产品信息
- 保持专业、友好和有帮助的语调
- 适时推进销售进程
- 以 <END_OF_TURN> 结尾

{name}：
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "name", "role", "company", "company_business", "experience",
                "specialization", "customer_context", "current_stage",
                "stage_description", "knowledge_context", "conversation_history"
            ]
        )

        return LLMChain(prompt=prompt, llm=self.llm, verbose=self.verbose)

    def _create_stage_analyzer_chain(self):
        """创建阶段分析链"""
        prompt_template = """
分析销售对话应该进入哪个阶段。

客户信息：{customer_context}
对话历史：{conversation_history}

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
            input_variables=["customer_context", "conversation_history"]
        )

        return LLMChain(prompt=prompt, llm=self.llm, verbose=False)

    def get_customer_context(self) -> str:
        """获取客户上下文信息"""
        if not self.customer_id:
            return "新客户，暂无档案信息"

        customer = self.customer_manager.get_customer(self.customer_id)
        if not customer:
            return "客户信息不存在"

        context = f"""
客户姓名：{customer.name}
公司：{customer.company}
职位：{customer.position}
行业：{customer.industry}
公司规模：{customer.company_size}
预算范围：{customer.budget_range}
痛点：{', '.join(customer.pain_points)}
兴趣点：{', '.join(customer.interests)}
当前状态：{customer.status.value}
        """
        return context.strip()

    def get_knowledge_context(self, user_input: str) -> str:
        """获取相关的产品知识"""
        if not user_input:
            return ""

        # 检查用户输入是否包含产品相关关键词
        product_keywords = ["产品", "解决方案", "价格", "功能", "服务", "技术", "系统", "平台"]

        if any(keyword in user_input for keyword in product_keywords):
            knowledge = self.knowledge_base.query(user_input)
            return f"相关产品信息：{knowledge}"

        return ""

    def analyze_stage(self, conversation_history: str) -> str:
        """分析当前对话阶段"""
        try:
            customer_context = self.get_customer_context()
            result = self.stage_analyzer_chain.invoke({
                "customer_context": customer_context,
                "conversation_history": conversation_history
            })
            stage = result.get("text", "1").strip()
            return stage if stage in SALES_STAGES else "1"
        except Exception as e:
            print(f"阶段分析错误: {e}")
            return "1"

    def step(self, user_input: str = None, channel: InteractionChannel = InteractionChannel.CHAT) -> str:
        """执行一步对话"""
        self.current_channel = channel

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
            self.current_stage = self.analyze_stage(history_str)

        # 获取客户上下文
        customer_context = self.get_customer_context()

        # 生成回复
        try:
            result = self.conversation_chain.invoke({
                **self.salesperson_info,
                "customer_context": customer_context,
                "current_stage": self.current_stage,
                "stage_description": SALES_STAGES[self.current_stage],
                "knowledge_context": knowledge_context,
                "conversation_history": history_str
            })

            response = result.get("text", "").strip()
            self.conversation_history.append(f"{self.salesperson_info['name']}：{response}<END_OF_TURN>")

            # 记录交互
            if self.customer_id and user_input:
                interaction = SalesInteraction(
                    interaction_id=f"INT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    customer_id=self.customer_id,
                    timestamp=datetime.datetime.now().isoformat(),
                    channel=channel,
                    stage=self.current_stage,
                    content=user_input,
                    outcome=response,
                    next_action="继续跟进",
                    salesperson=self.salesperson_info["name"]
                )
                self.customer_manager.add_interaction(interaction)

            return response

        except Exception as e:
            print(f"生成回复时出错: {e}")
            return "抱歉，我遇到了技术问题，请稍后再试。"

    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """获取全面的对话摘要"""
        summary = {
            "conversation_info": {
                "current_stage": self.current_stage,
                "stage_description": SALES_STAGES[self.current_stage],
                "conversation_turns": len(self.conversation_history),
                "channel": self.current_channel.value
            },
            "salesperson_info": self.salesperson_info,
            "customer_info": {},
            "analytics": {}
        }

        if self.customer_id:
            customer_insights = self.analytics.get_customer_insights(self.customer_id)
            summary["customer_info"] = customer_insights

        sales_summary = self.analytics.get_sales_summary()
        summary["analytics"] = sales_summary

        return summary

    def save_session(self):
        """保存会话数据"""
        self.customer_manager.save_data()

def demonstrate_enterprise_features():
    """演示企业级功能"""
    print("=" * 70)
    print("企业版 SalesGPT v5.0 功能演示")
    print("=" * 70)

    # 创建企业级销售代理
    sales_agent = EnterpriseSalesGPT(llm, customer_id="CUST001", verbose=False)

    print("\n1. 客户档案管理演示")
    print("-" * 40)
    customer = sales_agent.customer_manager.get_customer("CUST001")
    if customer:
        print(f"客户姓名: {customer.name}")
        print(f"公司: {customer.company}")
        print(f"行业: {customer.industry}")
        print(f"状态: {customer.status.value}")
        print(f"痛点: {', '.join(customer.pain_points)}")

    print("\n2. 销售数据分析演示")
    print("-" * 40)
    analytics_summary = sales_agent.analytics.get_sales_summary()
    print(f"总客户数: {analytics_summary['total_customers']}")
    print(f"总交互数: {analytics_summary['total_interactions']}")
    print(f"转化率: {analytics_summary['conversion_rate']:.1f}%")

    print("\n3. 个性化销售对话演示")
    print("-" * 40)

    # 开始对话
    response = sales_agent.step()
    print(f"\n{sales_agent.salesperson_info['name']}: {response}")

    # 模拟客户对话
    customer_inputs = [
        "你好，我想了解你们的解决方案",
        "我们是制造业，想要提升生产效率",
        "具体有什么产品可以帮助我们？",
        "价格大概是什么范围？"
    ]

    for customer_input in customer_inputs:
        print(f"\n客户: {customer_input}")
        response = sales_agent.step(customer_input, InteractionChannel.CHAT)
        print(f"{sales_agent.salesperson_info['name']}: {response}")

    print("\n4. 客户洞察分析")
    print("-" * 40)
    insights = sales_agent.analytics.get_customer_insights("CUST001")
    if insights:
        print(f"参与度评分: {insights['engagement_score']}")
        print(f"推荐行动: {', '.join(insights['recommended_actions'])}")

    # 保存会话数据
    sales_agent.save_session()
    print("\n✅ 会话数据已保存")

def main():
    """主函数"""
    print("企业版 SalesGPT v5.0")
    print("完整的企业级销售代理系统")

    try:
        # 演示企业级功能
        demonstrate_enterprise_features()

    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"\n\n程序出错: {e}")

    finally:
        print("\n" + "=" * 70)
        print("企业版 SalesGPT v5.0 演示结束")
        print("=" * 70)

if __name__ == "__main__":
    main()
