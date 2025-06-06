"""
Shell Tool 高级示例 - 使用现代 LangChain API
==========================================

本示例展示如何使用 LangChain 的 Shell 工具来执行系统命令，包括：
1. 基础命令执行
2. 跨平台兼容性处理
3. 安全命令过滤
4. 与智能代理集成
5. 实际应用场景

注意事项：
- Shell 工具默认没有安全防护，使用时需要谨慎
- 建议只在受控环境中使用
- 可以通过参数限制可执行的命令
- 生产环境中应该实现命令白名单

更新说明：
- 使用 langchain_community.tools 替代已弃用的 langchain.tools
- 添加了安全提示和错误处理
- 集成智能代理进行命令执行
- 添加了实际应用场景示例

作者：AI助手
日期：2024年
版本：2.0
"""

import os
import platform
import warnings
from typing import List

# LangChain 相关导入
try:
    from langchain_community.tools import ShellTool
    SHELL_TOOL_AVAILABLE = True
except ImportError:
    try:
        from langchain_experimental.tools import ShellTool
        SHELL_TOOL_AVAILABLE = True
        print("注意: 使用 langchain_experimental 中的 ShellTool")
        print("如果遇到问题，请运行: pip install langchain-experimental")
    except ImportError:
        SHELL_TOOL_AVAILABLE = False
        print("错误: 无法导入 ShellTool")
        print("请安装依赖: pip install langchain-experimental")

from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
import dotenv
import subprocess

# 加载环境变量
dotenv.load_dotenv()

# 过滤掉 Shell 工具的安全警告（如果你了解风险）
warnings.filterwarnings("ignore", message="The shell tool has no safeguards by default")

class SafeShellTool:
    """
    安全的 Shell 工具包装器
    提供命令白名单和安全检查功能
    """

    def __init__(self, allowed_commands: List[str] = None):
        """
        初始化安全 Shell 工具

        Args:
            allowed_commands: 允许执行的命令列表，如果为 None 则允许所有命令
        """
        if SHELL_TOOL_AVAILABLE:
            self.shell_tool = ShellTool()
            self.use_langchain_shell = True
        else:
            self.shell_tool = None
            self.use_langchain_shell = False
            print("警告: 使用 subprocess 作为备选方案")

        self.allowed_commands = allowed_commands or []
        self.system = platform.system()

        # 默认安全命令列表
        if not self.allowed_commands:
            self.allowed_commands = self._get_default_safe_commands()

    def _get_default_safe_commands(self) -> List[str]:
        """获取默认的安全命令列表"""
        if self.system == "Windows":
            return ["echo", "dir", "type", "find", "findstr", "where", "whoami", "date", "time", "start"]
        else:
            return ["echo", "ls", "cat", "grep", "find", "whoami", "date", "pwd", "which"]

    def _is_command_safe(self, command: str) -> bool:
        """
        检查命令是否安全

        Args:
            command: 要检查的命令

        Returns:
            bool: 命令是否安全
        """
        # 提取命令的第一个词（实际命令）
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False

        base_command = cmd_parts[0]

        # 检查是否在允许列表中
        return base_command in self.allowed_commands

    def run(self, command: str) -> str:
        """
        安全执行命令

        Args:
            command: 要执行的命令

        Returns:
            str: 命令执行结果
        """
        if not self._is_command_safe(command):
            return f"错误: 命令 '{command}' 不在安全命令列表中。允许的命令: {', '.join(self.allowed_commands)}"

        try:
            if self.use_langchain_shell and self.shell_tool:
                # 使用 LangChain ShellTool
                result = self.shell_tool.run({"commands": [command]})
                return result
            else:
                # 使用 subprocess 作为备选方案
                return self._run_with_subprocess(command)
        except Exception as e:
            return f"命令执行错误: {str(e)}"

    def _run_with_subprocess(self, command: str) -> str:
        """
        使用 subprocess 执行命令

        Args:
            command: 要执行的命令

        Returns:
            str: 命令执行结果
        """
        try:
            if self.system == "Windows":
                # Windows 系统使用 cmd
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            else:
                # Unix/Linux 系统使用 bash
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"命令执行失败 (返回码: {result.returncode}): {result.stderr.strip()}"

        except subprocess.TimeoutExpired:
            return "命令执行超时 (30秒)"
        except Exception as e:
            return f"subprocess 执行错误: {str(e)}"

def demonstrate_basic_shell_usage():
    """演示基础 Shell 工具使用"""
    print("=" * 60)
    print("1. 基础 Shell 工具使用示例")
    print("=" * 60)

    # 创建安全的 Shell 工具
    safe_shell = SafeShellTool()

    # 跨平台命令示例
    if platform.system() == "Windows":
        commands = [
            "echo Hello from Windows!",
            "dir /b",
            "whoami",
            "date /t"
        ]
    else:
        commands = [
            "echo 'Hello from Unix/Linux!'",
            "ls -la",
            "whoami",
            "date"
        ]

    for i, cmd in enumerate(commands, 1):
        print(f"\n{i}. 执行命令: {cmd}")
        result = safe_shell.run(cmd)
        print(f"结果:\n{result}")

def demonstrate_browser_automation():
    """演示浏览器自动化功能"""
    print("\n" + "=" * 60)
    print("2. 浏览器自动化示例")
    print("=" * 60)

    safe_shell = SafeShellTool()

    if platform.system() == "Windows":
        print("\n在 Windows 系统上打开浏览器:")

        # 基础浏览器启动
        print("1. 启动默认浏览器:")
        result = safe_shell.run("start chrome.exe")
        print(f"结果: {result}")

        # 打开特定网页
        print("\n2. 打开百度搜索 GPT4:")
        result = safe_shell.run("start chrome.exe http://www.baidu.com/s?wd=gpt4")
        print(f"结果: {result}")

        # 打开多个网页
        print("\n3. 打开多个有用的网页:")
        urls = [
            "https://python.langchain.com/",
            "https://github.com/langchain-ai/langchain",
            "https://docs.python.org/3/"
        ]

        for i, url in enumerate(urls, 1):
            print(f"   {i}. 打开: {url}")
            result = safe_shell.run(f"start chrome.exe {url}")
            print(f"   结果: {result}")
    else:
        print("在 Unix/Linux 系统上，浏览器启动命令可能不同")
        print("可以尝试: xdg-open, firefox, chromium-browser 等命令")

def create_shell_agent():
    """创建集成 Shell 工具的智能代理"""
    print("\n" + "=" * 60)
    print("3. Shell 工具与智能代理集成")
    print("=" * 60)

    # 检查 API 密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("警告: 未找到 OPENAI_API_KEY，跳过代理示例")
        print("请在 .env 文件中设置 OPENAI_API_KEY")
        return None

    # 创建安全的 Shell 工具
    safe_shell = SafeShellTool()

    # 将 SafeShellTool 包装为 LangChain Tool
    shell_tool = Tool(
        name="shell_executor",
        description="执行安全的系统命令。只能执行预定义的安全命令列表中的命令。",
        func=safe_shell.run
    )

    # 创建 LLM
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1/",
        model="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.1
    )

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个系统管理助手，可以帮助用户执行安全的系统命令。

你有以下工具可用：
- shell_executor: 执行安全的系统命令

安全规则：
1. 只执行安全命令列表中的命令
2. 不执行可能危险的操作（如删除文件、修改系统配置等）
3. 如果用户请求不安全的操作，请解释为什么不能执行并建议替代方案

请用中文回复用户。"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # 创建代理
    agent = create_openai_functions_agent(llm, [shell_tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[shell_tool], verbose=True)

    return agent_executor

def demonstrate_agent_usage(agent_executor):
    """演示代理使用"""
    if not agent_executor:
        return

    # 测试查询列表
    queries = [
        "帮我查看当前目录下的文件",
        "显示当前用户名",
        "获取当前系统时间",
        "打开浏览器并搜索 LangChain",
        "删除所有文件",  # 这个应该被拒绝
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. 用户查询: {query}")
        print("-" * 40)
        try:
            result = agent_executor.invoke({"input": query})
            print(f"代理回复: {result['output']}")
        except Exception as e:
            print(f"执行错误: {e}")

def demonstrate_practical_scenarios():
    """演示实际应用场景"""
    print("\n" + "=" * 60)
    print("4. 实际应用场景示例")
    print("=" * 60)

    safe_shell = SafeShellTool()

    scenarios = [
        {
            "name": "系统信息收集",
            "description": "收集基本系统信息",
            "commands": ["whoami", "date"] if platform.system() != "Windows" else ["whoami", "date /t"]
        },
        {
            "name": "文件系统探索",
            "description": "安全地探索文件系统",
            "commands": ["ls -la"] if platform.system() != "Windows" else ["dir"]
        },
        {
            "name": "文本处理",
            "description": "简单的文本处理操作",
            "commands": ["echo 'LangChain Shell Tool Demo'"]
        },
        {
            "name": "开发者工具",
            "description": "常用的开发者命令",
            "commands": ["echo 'Current working directory:'", "pwd"] if platform.system() != "Windows" else ["echo Current working directory:", "cd"]
        }
    ]

    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        print(f"描述: {scenario['description']}")
        print("-" * 30)

        for cmd in scenario['commands']:
            print(f"执行: {cmd}")
            result = safe_shell.run(cmd)
            print(f"结果: {result}")
            print()

def demonstrate_error_handling():
    """演示错误处理"""
    print("\n" + "=" * 60)
    print("5. 错误处理和安全检查示例")
    print("=" * 60)

    safe_shell = SafeShellTool()

    # 测试不安全的命令
    unsafe_commands = [
        "rm -rf /",  # 危险的删除命令
        "del *.*",   # Windows 删除命令
        "format c:", # 格式化命令
        "shutdown -s -t 0",  # 关机命令
        "python -c 'import os; os.system(\"rm -rf /\")'",  # 通过 Python 执行危险命令
    ]

    print("测试不安全命令的拦截:")
    for i, cmd in enumerate(unsafe_commands, 1):
        print(f"\n{i}. 尝试执行危险命令: {cmd}")
        result = safe_shell.run(cmd)
        print(f"安全检查结果: {result}")

def main():
    """主函数"""
    print("LangChain Shell Tool 高级示例")
    print("支持安全命令执行、智能代理集成和实际应用场景")

    try:
        # 1. 基础使用演示
        demonstrate_basic_shell_usage()

        # 2. 浏览器自动化演示
        demonstrate_browser_automation()

        # 3. 智能代理集成演示
        agent_executor = create_shell_agent()
        demonstrate_agent_usage(agent_executor)

        # 4. 实际应用场景演示
        demonstrate_practical_scenarios()

        # 5. 错误处理演示
        demonstrate_error_handling()

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n程序执行出错: {e}")

    finally:
        print("\n" + "=" * 60)
        print("安全提示:")
        print("- Shell 工具功能强大但需要谨慎使用")
        print("- 生产环境中应该实现严格的命令白名单")
        print("- 建议对用户输入进行验证和清理")
        print("- 考虑使用容器或沙箱环境来隔离命令执行")
        print("- 定期审查和更新安全策略")
        print("=" * 60)

if __name__ == "__main__":
    main()
