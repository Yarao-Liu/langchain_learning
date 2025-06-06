"""
知识库版 SalesGPT v3.0 - 集成简单知识库系统
==========================================

这是SalesGPT系列的第三个版本，在增强对话版基础上增加了：
1. 基于关键词的知识库系统
2. 产品信息管理
3. 智能信息检索
4. 上下文感知回复
5. 结构化产品数据

功能特点：
- 无需复杂依赖的知识库
- 快速关键词匹配
- 丰富的产品信息
- 智能信息推荐
- 上下文相关回复

作者：AI助手
日期：2024年
版本：3.0 - 知识库版
"""

import os
import warnings
import json
from typing import Dict, Any, List

import dotenv
from langchain.chains.llm import LLMChain
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

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

class SimpleKnowledgeBase:
    """简单的基于关键词的知识库"""
    
    def __init__(self):
        """初始化知识库"""
        self.knowledge_data = {
            "产品信息": {
                "智能办公系统": {
                    "价格": "智能办公系统价格：基础版8万元/年，专业版15万元/年，企业版25万元/年",
                    "功能": "包括智能文档管理、自动化工作流、AI助手、数据分析、团队协作等功能",
                    "优势": "提升工作效率30%，减少重复工作，智能决策支持，无缝集成现有系统",
                    "适用": "适用于50-1000人的企业，特别是知识密集型行业",
                    "实施": "实施周期2-4周，提供完整培训和技术支持"
                },
                "AI数据分析平台": {
                    "价格": "AI数据分析平台：标准版12万元/年，高级版20万元/年，定制版面议",
                    "功能": "实时数据处理、智能报表生成、预测分析、可视化展示、API接口",
                    "优势": "快速洞察业务趋势，自动化报告生成，预测性分析，降低决策风险",
                    "适用": "适用于需要数据驱动决策的企业，如电商、金融、制造业",
                    "实施": "实施周期3-6周，包含数据迁移和系统集成"
                },
                "智能客服机器人": {
                    "价格": "智能客服机器人：基础版5万元/年，增强版10万元/年，定制版15万元/年",
                    "功能": "24小时在线服务、多渠道接入、智能问答、情感分析、人工转接",
                    "优势": "降低客服成本60%，提升响应速度，改善客户体验，数据驱动优化",
                    "适用": "适用于有大量客户咨询的企业，如电商、金融、教育行业",
                    "实施": "实施周期1-2周，快速部署上线"
                }
            },
            "服务信息": {
                "技术支持": "提供7×24小时技术支持，专业工程师团队，远程协助和现场服务",
                "培训服务": "提供完整的用户培训，包括操作培训、管理培训和高级功能培训",
                "定制开发": "根据客户特殊需求提供定制开发服务，确保系统完美匹配业务流程",
                "数据迁移": "专业的数据迁移服务，确保数据安全和完整性，零停机迁移",
                "系统集成": "与现有系统无缝集成，支持主流ERP、CRM、OA系统对接"
            },
            "公司优势": {
                "技术实力": "拥有50+技术专家，10年AI技术积累，多项核心技术专利",
                "客户案例": "服务500+企业客户，包括多家世界500强企业，客户满意度98%",
                "行业经验": "深耕制造、金融、教育、电商等多个行业，丰富的项目经验",
                "服务保障": "完善的售后服务体系，快速响应，持续优化升级"
            }
        }
        
        # 关键词映射
        self.keyword_mapping = {
            "价格": ["价格", "费用", "成本", "多少钱", "报价"],
            "功能": ["功能", "特性", "能力", "作用", "用途"],
            "优势": ["优势", "好处", "价值", "效果", "收益"],
            "适用": ["适用", "适合", "行业", "企业", "客户"],
            "实施": ["实施", "部署", "上线", "周期", "时间"],
            "技术支持": ["支持", "服务", "维护", "帮助"],
            "培训服务": ["培训", "学习", "教学", "指导"],
            "定制开发": ["定制", "开发", "个性化", "特殊需求"],
            "数据迁移": ["迁移", "导入", "数据", "转移"],
            "系统集成": ["集成", "对接", "整合", "兼容"],
            "技术实力": ["技术", "实力", "专家", "团队"],
            "客户案例": ["案例", "客户", "成功", "经验"],
            "行业经验": ["行业", "经验", "专业", "领域"],
            "服务保障": ["保障", "承诺", "质量", "可靠"]
        }
        
        # 产品关键词
        self.product_keywords = {
            "智能办公系统": ["办公", "OA", "协作", "文档", "工作流"],
            "AI数据分析平台": ["数据", "分析", "报表", "BI", "预测"],
            "智能客服机器人": ["客服", "机器人", "聊天", "问答", "服务"]
        }
    
    def search_knowledge(self, query: str) -> str:
        """基于关键词搜索知识"""
        query_lower = query.lower()
        results = []
        
        # 查找相关产品
        target_products = []
        for product, keywords in self.product_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                target_products.append(product)
        
        if not target_products:
            target_products = list(self.knowledge_data["产品信息"].keys())
        
        # 查找相关信息类型
        for info_type, keywords in self.keyword_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                # 在产品信息中查找
                for product in target_products:
                    if product in self.knowledge_data["产品信息"]:
                        if info_type in self.knowledge_data["产品信息"][product]:
                            results.append(f"{product} - {self.knowledge_data['产品信息'][product][info_type]}")
                
                # 在服务信息中查找
                if info_type in self.knowledge_data["服务信息"]:
                    results.append(self.knowledge_data["服务信息"][info_type])
                
                # 在公司优势中查找
                if info_type in self.knowledge_data["公司优势"]:
                    results.append(self.knowledge_data["公司优势"][info_type])
        
        if results:
            return " | ".join(results[:3])  # 返回前3个最相关的结果
        else:
            return "我需要更多信息来为您提供准确的回答，请您具体说明想了解什么？"

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

class KnowledgeBasedSalesGPT:
    """知识库版销售对话代理"""
    
    def __init__(self, llm, verbose=True):
        """初始化销售代理"""
        self.llm = llm
        self.verbose = verbose
        self.knowledge_base = SimpleKnowledgeBase()
        self.stage_analyzer = StageAnalyzer(llm)
        self.conversation_history = []
        self.current_stage = "1"
        
        # 销售人员信息
        self.salesperson_info = {
            "name": "小李",
            "role": "高级解决方案顾问",
            "company": "智能科技有限公司",
            "company_business": "专注于为企业提供AI驱动的智能化解决方案，包括智能办公、数据分析、客服机器人等产品",
            "company_values": "以技术创新为驱动，为客户创造价值，推动企业数字化转型",
            "contact_purpose": "了解客户的数字化需求，为客户提供最适合的智能化解决方案"
        }
        
        # 创建对话链
        self.conversation_chain = self._create_conversation_chain()
    
    def _create_conversation_chain(self):
        """创建知识增强的对话链"""
        prompt_template = """
你是{name}，{role}，在{company}工作。

公司业务：{company_business}
公司价值观：{company_values}
联系目的：{contact_purpose}

当前销售阶段：{current_stage}
阶段说明：{stage_description}

相关产品知识：
{knowledge_context}

对话历史：
{conversation_history}

请根据当前阶段和产品知识生成专业回复：
- 充分利用提供的产品知识
- 根据客户问题提供准确信息
- 保持专业和友好的语调
- 适时推进销售进程
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
        return self.knowledge_base.search_knowledge(user_input)
    
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
            "company": self.salesperson_info["company"]
        }

def demonstrate_knowledge_search():
    """演示知识库搜索功能"""
    print("=" * 60)
    print("知识库搜索功能演示")
    print("=" * 60)
    
    kb = SimpleKnowledgeBase()
    
    test_queries = [
        "智能办公系统多少钱？",
        "数据分析平台有什么功能？",
        "客服机器人的优势是什么？",
        "你们提供什么技术支持？",
        "有哪些成功案例？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 问题: {query}")
        answer = kb.search_knowledge(query)
        print(f"   回答: {answer}")

def demonstrate_knowledge_sales():
    """演示知识库版销售对话"""
    print("\n" + "=" * 60)
    print("知识库版销售对话演示")
    print("=" * 60)
    
    sales_agent = KnowledgeBasedSalesGPT(llm, verbose=False)
    
    print(f"\n{sales_agent.salesperson_info['name']}开始对话...")
    response = sales_agent.step()
    print(f"\n{sales_agent.salesperson_info['name']}: {response}")
    
    # 模拟客户对话
    customer_inputs = [
        "你好，我想了解你们的产品",
        "我们公司想要提升办公效率，有什么解决方案？",
        "智能办公系统的价格怎么样？",
        "具体有哪些功能？",
        "实施起来复杂吗？需要多长时间？"
    ]
    
    for customer_input in customer_inputs:
        print(f"\n客户: {customer_input}")
        response = sales_agent.step(customer_input)
        print(f"{sales_agent.salesperson_info['name']}: {response}")
        
        summary = sales_agent.get_conversation_summary()
        print(f"[阶段 {summary['current_stage']}: {summary['stage_description']}]")

def main():
    """主函数"""
    print("知识库版 SalesGPT v3.0")
    print("集成简单知识库系统")
    
    try:
        # 1. 演示知识库搜索
        demonstrate_knowledge_search()
        
        # 2. 演示知识库版销售对话
        demonstrate_knowledge_sales()
        
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"\n\n程序出错: {e}")
    
    finally:
        print("\n" + "=" * 60)
        print("知识库版 SalesGPT v3.0 演示结束")
        print("=" * 60)

if __name__ == "__main__":
    main()
