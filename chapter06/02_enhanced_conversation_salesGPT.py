"""
增强对话版 SalesGPT v2.0 - 改进的对话管理系统
=============================================

这是SalesGPT系列的第二个版本，在基础版基础上增加了：
1. 智能的阶段分析系统
2. 更丰富的销售人员信息
3. 改进的对话管理
4. 更好的错误处理
5. 对话状态跟踪

功能特点：
- 使用LLM进行阶段分析
- 更详细的销售流程
- 改进的提示词工程
- 对话上下文管理
- 销售策略指导

作者：AI助手
日期：2024年
版本：2.0 - 增强对话版
"""

import os
import warnings
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

# 详细的销售对话阶段定义
SALES_STAGES = {
    "1": "介绍：通过介绍自己和公司来开始对话，建立信任关系",
    "2": "资格确认：确认客户是否是合适的潜在客户，了解决策权限",
    "3": "需求分析：深入了解客户的具体需求、痛点和期望",
    "4": "价值主张：展示产品/服务的独特价值和竞争优势",
    "5": "解决方案展示：根据客户需求展示具体的解决方案",
    "6": "异议处理：处理客户的疑虑、反对意见和担忧",
    "7": "成交推进：推进销售进程，提出具体的下一步行动"
}

class StageAnalyzer:
    """智能阶段分析器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.analyzer_chain = self._create_analyzer_chain()
    
    def _create_analyzer_chain(self):
        """创建阶段分析链"""
        prompt_template = """
你是一个销售对话阶段分析专家。根据对话历史，判断销售对话应该进入哪个阶段。

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

请分析对话内容，选择最合适的下一个阶段。
只回答数字1-7，不要添加其他内容。
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
            
            # 验证返回的阶段是否有效
            if stage in SALES_STAGES:
                return stage
            else:
                return "1"  # 默认返回介绍阶段
                
        except Exception as e:
            print(f"阶段分析错误: {e}")
            return "1"

class EnhancedSalesGPT:
    """增强版销售对话代理"""
    
    def __init__(self, llm, verbose=True):
        """初始化销售代理"""
        self.llm = llm
        self.verbose = verbose
        self.stage_analyzer = StageAnalyzer(llm)
        self.conversation_history = []
        self.current_stage = "1"
        
        # 详细的销售人员信息
        self.salesperson_info = {
            "name": "小陈",
            "role": "高级销售顾问",
            "company": "创新科技有限公司",
            "company_business": "专注于为企业提供智能化解决方案，包括AI产品、自动化系统和数字化转型服务",
            "company_values": "以客户为中心，提供高质量的产品和服务，帮助客户实现业务目标",
            "contact_purpose": "了解客户的业务需求，为客户提供最适合的智能化解决方案",
            "years_experience": "5年销售经验"
        }
        
        # 创建对话链
        self.conversation_chain = self._create_conversation_chain()
    
    def _create_conversation_chain(self):
        """创建增强的对话链"""
        prompt_template = """
你是{name}，一名{role}，在{company}工作了{years_experience}。

公司业务：{company_business}
公司价值观：{company_values}
联系客户目的：{contact_purpose}

当前销售阶段：{current_stage}
阶段说明：{stage_description}

对话历史：
{conversation_history}

销售策略指导：
- 根据当前阶段调整对话策略
- 保持专业、友好和有帮助的语调
- 主动倾听客户需求
- 提供有价值的信息
- 适时推进销售进程

请根据当前阶段生成合适的回复，以 <END_OF_TURN> 结尾：

{name}：
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "name", "role", "company", "company_business", "company_values",
                "contact_purpose", "years_experience", "current_stage", 
                "stage_description", "conversation_history"
            ]
        )
        
        return LLMChain(prompt=prompt, llm=self.llm, verbose=self.verbose)
    
    def step(self, user_input: str = None) -> str:
        """执行一步对话"""
        # 如果有用户输入，添加到历史记录
        if user_input:
            self.conversation_history.append(f"客户：{user_input}<END_OF_TURN>")
        
        # 构建对话历史字符串
        history_str = "".join(self.conversation_history[-10:])  # 保留最近10轮对话
        if not history_str:
            history_str = "对话开始"
        
        # 分析当前阶段
        if len(self.conversation_history) > 0:  # 有对话历史时才进行阶段分析
            self.current_stage = self.stage_analyzer.analyze_stage(history_str)
        
        # 生成回复
        try:
            result = self.conversation_chain.invoke({
                **self.salesperson_info,
                "current_stage": self.current_stage,
                "stage_description": SALES_STAGES[self.current_stage],
                "conversation_history": history_str
            })
            
            response = result.get("text", "").strip()
            
            # 添加到历史记录
            self.conversation_history.append(f"{self.salesperson_info['name']}：{response}<END_OF_TURN>")
            
            return response
            
        except Exception as e:
            print(f"生成回复时出错: {e}")
            return "抱歉，我遇到了一些技术问题，请稍后再试。"
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        return {
            "current_stage": self.current_stage,
            "stage_description": SALES_STAGES[self.current_stage],
            "conversation_turns": len(self.conversation_history),
            "salesperson": self.salesperson_info["name"],
            "company": self.salesperson_info["company"]
        }
    
    def reset_conversation(self):
        """重置对话状态"""
        self.conversation_history = []
        self.current_stage = "1"

def demonstrate_enhanced_sales():
    """演示增强版销售对话"""
    print("=" * 60)
    print("增强对话版 SalesGPT v2.0 演示")
    print("=" * 60)
    
    # 创建销售代理
    sales_agent = EnhancedSalesGPT(llm, verbose=False)
    
    print(f"\n{sales_agent.salesperson_info['name']}开始对话...")
    
    # 开始对话
    response = sales_agent.step()
    print(f"\n{sales_agent.salesperson_info['name']}: {response}")
    
    # 模拟更复杂的客户对话
    customer_inputs = [
        "你好，我想了解一下你们公司的产品",
        "我们是一家制造企业，想要提升生产效率",
        "具体有什么解决方案吗？价格怎么样？",
        "听起来不错，但我担心实施起来会很复杂",
        "需要多长时间才能看到效果？",
        "我需要和团队讨论一下，你能提供更详细的资料吗？"
    ]
    
    for customer_input in customer_inputs:
        print(f"\n客户: {customer_input}")
        response = sales_agent.step(customer_input)
        print(f"{sales_agent.salesperson_info['name']}: {response}")
        
        # 显示对话摘要
        summary = sales_agent.get_conversation_summary()
        print(f"[阶段 {summary['current_stage']}: {summary['stage_description']}]")

def interactive_enhanced_demo():
    """交互式增强演示"""
    print("\n" + "=" * 60)
    print("交互式增强对话演示")
    print("=" * 60)
    print("输入 'quit' 退出，'summary' 查看对话摘要，'reset' 重置对话")
    print("-" * 60)
    
    sales_agent = EnhancedSalesGPT(llm, verbose=False)
    
    # 开始对话
    print(f"\n{sales_agent.salesperson_info['name']}开始对话...")
    response = sales_agent.step()
    print(f"\n{sales_agent.salesperson_info['name']}: {response}")
    
    while True:
        try:
            user_input = input(f"\n您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n对话结束，感谢您的时间！")
                break
            
            if user_input.lower() == 'summary':
                summary = sales_agent.get_conversation_summary()
                print(f"\n对话摘要:")
                print(f"- 当前阶段: {summary['current_stage']} - {summary['stage_description']}")
                print(f"- 对话轮数: {summary['conversation_turns']}")
                print(f"- 销售顾问: {summary['salesperson']} ({summary['company']})")
                continue
            
            if user_input.lower() == 'reset':
                sales_agent.reset_conversation()
                print("\n对话已重置")
                response = sales_agent.step()
                print(f"\n{sales_agent.salesperson_info['name']}: {response}")
                continue
            
            if not user_input:
                continue
            
            response = sales_agent.step(user_input)
            print(f"\n{sales_agent.salesperson_info['name']}: {response}")
            
        except KeyboardInterrupt:
            print("\n\n对话被中断")
            break
        except Exception as e:
            print(f"\n出现错误: {e}")

def main():
    """主函数"""
    print("增强对话版 SalesGPT v2.0")
    print("智能阶段分析 + 增强对话管理")
    
    try:
        # 1. 演示增强版销售对话
        demonstrate_enhanced_sales()
        
        # 2. 交互式演示
        choice = input("\n是否要进行交互式对话？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_enhanced_demo()
    
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"\n\n程序出错: {e}")
    
    finally:
        print("\n" + "=" * 60)
        print("增强对话版 SalesGPT v2.0 演示结束")
        print("=" * 60)

if __name__ == "__main__":
    main()
