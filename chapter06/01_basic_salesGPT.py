"""
基础版 SalesGPT v1.0 - 最简单的销售对话代理
===============================================

这是SalesGPT系列的第一个版本，提供最基础的销售对话功能：
1. 简单的对话阶段管理
2. 基础的销售流程
3. 简单的用户交互
4. 基本的提示词模板

功能特点：
- 轻量级实现
- 易于理解和修改
- 基础的销售对话流程
- 简单的阶段转换逻辑

作者：AI助手
日期：2024年
版本：1.0 - 基础版
"""

import os
import warnings
from typing import Dict, Any

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
    "1": "介绍：开始对话，介绍自己和公司",
    "2": "了解需求：询问客户的基本需求",
    "3": "产品介绍：介绍产品的基本信息",
    "4": "解答疑问：回答客户的问题",
    "5": "推进成交：尝试促成交易"
}

class BasicSalesGPT:
    """基础版销售对话代理"""
    
    def __init__(self, llm):
        """初始化销售代理"""
        self.llm = llm
        self.current_stage = "1"
        self.conversation_history = []
        
        # 基础销售人员信息
        self.salesperson_info = {
            "name": "小王",
            "role": "销售顾问",
            "company": "科技公司",
            "product": "智能产品"
        }
        
        # 创建对话链
        self.conversation_chain = self._create_conversation_chain()
    
    def _create_conversation_chain(self):
        """创建对话链"""
        prompt_template = """
你是{name}，一名专业的{role}，在{company}工作。

当前销售阶段：{stage}
对话历史：{history}

请根据当前阶段生成合适的回复：
- 保持专业和友好的语调
- 根据销售阶段调整对话策略
- 回复要简洁明了
- 以 <END_OF_TURN> 结尾

{name}：
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["name", "role", "company", "stage", "history"]
        )
        
        return LLMChain(prompt=prompt, llm=self.llm, verbose=False)
    
    def _get_next_stage(self, user_input: str) -> str:
        """简单的阶段转换逻辑"""
        current_stage_num = int(self.current_stage)
        
        # 简单的关键词匹配来决定下一阶段
        if current_stage_num == 1:  # 介绍阶段
            if any(word in user_input.lower() for word in ["你好", "想了解", "需要"]):
                return "2"
        elif current_stage_num == 2:  # 了解需求阶段
            if any(word in user_input.lower() for word in ["价格", "功能", "产品"]):
                return "3"
        elif current_stage_num == 3:  # 产品介绍阶段
            if any(word in user_input.lower() for word in ["问题", "疑问", "担心"]):
                return "4"
        elif current_stage_num == 4:  # 解答疑问阶段
            if any(word in user_input.lower() for word in ["考虑", "购买", "决定"]):
                return "5"
        
        # 如果没有匹配的关键词，保持当前阶段或递进
        if current_stage_num < 5:
            return str(current_stage_num + 1)
        else:
            return "5"  # 保持在最后阶段
    
    def step(self, user_input: str = None) -> str:
        """执行一步对话"""
        # 如果有用户输入，添加到历史记录并更新阶段
        if user_input:
            self.conversation_history.append(f"客户：{user_input}")
            self.current_stage = self._get_next_stage(user_input)
        
        # 构建对话历史字符串
        history_str = "\n".join(self.conversation_history[-5:])  # 只保留最近5轮对话
        if not history_str:
            history_str = "对话开始"
        
        # 生成回复
        try:
            result = self.conversation_chain.invoke({
                "name": self.salesperson_info["name"],
                "role": self.salesperson_info["role"],
                "company": self.salesperson_info["company"],
                "stage": SALES_STAGES[self.current_stage],
                "history": history_str
            })
            
            response = result.get("text", "").strip()
            
            # 添加到历史记录
            self.conversation_history.append(f"{self.salesperson_info['name']}：{response}")
            
            return response
            
        except Exception as e:
            print(f"生成回复时出错: {e}")
            return "抱歉，我遇到了一些问题，请稍后再试。"
    
    def get_current_stage_info(self) -> Dict[str, str]:
        """获取当前阶段信息"""
        return {
            "stage_number": self.current_stage,
            "stage_description": SALES_STAGES[self.current_stage],
            "conversation_turns": len(self.conversation_history)
        }

def demonstrate_basic_sales():
    """演示基础销售对话"""
    print("=" * 50)
    print("基础版 SalesGPT v1.0 演示")
    print("=" * 50)
    
    # 创建销售代理
    sales_agent = BasicSalesGPT(llm)
    
    print(f"\n{sales_agent.salesperson_info['name']}开始对话...")
    
    # 开始对话
    response = sales_agent.step()
    print(f"\n{sales_agent.salesperson_info['name']}: {response}")
    
    # 模拟客户对话
    customer_inputs = [
        "你好，我想了解一下你们的产品",
        "价格怎么样？",
        "有什么特色功能吗？",
        "我有点担心质量问题",
        "让我考虑一下"
    ]
    
    for customer_input in customer_inputs:
        print(f"\n客户: {customer_input}")
        response = sales_agent.step(customer_input)
        print(f"{sales_agent.salesperson_info['name']}: {response}")
        
        # 显示当前阶段
        stage_info = sales_agent.get_current_stage_info()
        print(f"[阶段 {stage_info['stage_number']}: {stage_info['stage_description']}]")

def interactive_demo():
    """交互式演示"""
    print("\n" + "=" * 50)
    print("交互式对话演示")
    print("=" * 50)
    print("输入 'quit' 退出对话")
    print("-" * 50)
    
    sales_agent = BasicSalesGPT(llm)
    
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
            
            if not user_input:
                continue
            
            response = sales_agent.step(user_input)
            print(f"\n{sales_agent.salesperson_info['name']}: {response}")
            
            # 显示当前阶段
            stage_info = sales_agent.get_current_stage_info()
            print(f"[阶段 {stage_info['stage_number']}: {stage_info['stage_description']}]")
            
        except KeyboardInterrupt:
            print("\n\n对话被中断")
            break
        except Exception as e:
            print(f"\n出现错误: {e}")

def main():
    """主函数"""
    print("基础版 SalesGPT v1.0")
    print("最简单的销售对话代理")
    
    try:
        # 1. 演示基础销售对话
        demonstrate_basic_sales()
        
        # 2. 交互式演示
        choice = input("\n是否要进行交互式对话？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_demo()
    
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"\n\n程序出错: {e}")
    
    finally:
        print("\n" + "=" * 50)
        print("基础版 SalesGPT v1.0 演示结束")
        print("=" * 50)

if __name__ == "__main__":
    main()
