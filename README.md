# LangChain 中文学习实践项目

这是一个全面的 LangChain 框架中文学习项目，从基础连接到高级应用，系统性地展示了如何使用 LangChain 构建各种 AI 应用。项目包含 70+ 个实践示例，涵盖模型连接、提示工程、输出解析、RAG 系统、链式处理、智能代理和渐进式SalesGPT开发等核心功能。

## 项目特点

- 🚀 **完整的学习路径**：从基础到高级，循序渐进的 7 个章节
- 🌏 **中文优化**：专门针对中文场景优化，支持中文处理和问答
- 🔧 **多模型支持**：集成 OpenAI、智谱AI、Ollama 等多种模型服务
- 📚 **丰富的文档处理**：支持 PDF、HTML、文本、Markdown 等多种格式
- 🔍 **高效检索系统**：使用 FAISS、Chroma 向量数据库进行语义检索
- 🤖 **智能代理系统**：包含搜索、工具调用、记忆管理等高级功能
- 🎯 **渐进式开发**：SalesGPT系列展示从简单到企业级的完整开发过程
- 💡 **详细注释**：每个示例都有完整的中文注释和说明
- 🛠️ **实用工具**：包含自定义解析器、记忆管理、错误处理等实用组件

## 目录结构

```
.
├── chapter00/          # 模型连接基础
│   ├── 01_connect_invoke.py           # OpenAI API 连接调用
│   ├── 02_connect_stream.py           # OpenAI API 流式输出
│   ├── 03_connect_zhipu_invoke.py     # 智谱AI 连接调用
│   ├── 04_connect_zhipu_stream.py     # 智谱AI 流式输出
│   ├── 05_connect_custom.py           # 自定义模型连接
│   ├── 06_connect_ollama_invoke.py    # Ollama 本地模型调用
│   └── 07_connect_ollama_stream.py    # Ollama 流式输出
├── chapter01/          # 提示工程与模板
│   ├── 01_template_chat.py            # 聊天提示模板
│   ├── 02_template_fewshot.py         # 少样本学习模板
│   ├── 03_selector_maxMarginalRelevance.py  # 最大边际相关性选择器
│   ├── 04_selector_ngram.py           # N-gram 重叠选择器
│   ├── 05_template_fewshot.py         # 少样本提示模板进阶
│   ├── 06_selector_semanticSimilarity.py   # 语义相似度选择器
│   ├── 07_template_fewshot.py         # 少样本模板综合应用
│   ├── 08_similarity_chroma.py        # Chroma 向量相似度
│   ├── 09_template_nest.py            # 嵌套提示模板
│   ├── 10_template_date.py            # 日期处理模板
│   └── 11_template_full.py            # 完整提示模板示例
├── chapter02/          # 输出解析器
│   ├── 21_cache_inMemory.py           # 内存缓存
│   ├── 22_cache_inSQLite.py           # SQLite 缓存
│   ├── 23_parser_csv.py               # CSV 格式解析器
│   ├── 24_parser_date.py              # 日期时间解析器
│   ├── 25_parser_enum.py              # 枚举类型解析器
│   ├── 25_parser_json.py              # JSON 格式解析器
│   ├── 26_parser_xml.py               # XML 格式解析器
│   └── 27_parser_custom.py            # 自定义解析器
├── chapter03/          # RAG（检索增强生成）
│   ├── 31_rag_faiss.py                # FAISS 向量检索
│   ├── 32_rag_llm.py                  # LLM 集成检索
│   ├── 33_rag_llm2.py                 # 高级 LLM 检索
│   ├── 34_rag_unstructured_textloader.py      # 文本文件加载
│   ├── 35_rag_unstructured_DirectoryLoader.py # 目录批量加载
│   ├── 36_rag_unstructured_pdfloader.py       # PDF 文件加载
│   ├── 37_rag_unstructured_pdfDirectoryLoader.py # PDF 目录加载
│   ├── 38_rag_unstructured_htmlLoader.py      # HTML 文件加载
│   ├── 39_rag_unstructured_TextSpliter.py     # 文本分割器
│   ├── 40_rag_unstructured_markdown.py        # Markdown 处理
│   └── text/                          # 示例文档目录
│       ├── 01.txt                     # 示例文本文件
│       ├── 01.pdf                     # 示例PDF文件
│       └── 01.html                    # 示例HTML文件
├── chapter04/          # Runnable 链式处理
│   ├── 41_retriever_MultiVectorRetriever.py   # 多向量检索器
│   ├── 42_retriever_SelfQuerying.py           # 自查询检索器
│   ├── 43_runnable_serial.py                  # 串行处理链
│   ├── 44_runnable_parallel.py                # 并行处理链
│   ├── 45_runnable_passthrough.py             # 数据传递链
│   ├── 46_runnable_branch.py                  # 分支处理链
│   ├── 47_runnable_param.py                   # 参数处理链
│   ├── 48_runnable_config.py                  # 配置处理链
│   ├── 49_runnable_chain.py                   # 复合处理链
│   ├── 50_custom_generator.py                 # 自定义生成器
│   └── text/                                  # 测试文档
│       ├── ai_introduction.txt                # AI 介绍文档
│       └── machine_learning.txt               # 机器学习文档
├── chapter05/          # 智能代理系统
│   ├── 51_agent_serpApi.py            # SerpAPI 搜索代理
│   ├── 52_agent_tool.py               # 工具集成代理
│   ├── 53_agent_memory.py             # 记忆管理代理
│   ├── 54_advanced_memory.py          # 高级记忆管理
│   ├── 55_agent_json.py               # JSON 格式代理
│   ├── 55_agent_xml.py                # XML 格式代理
│   ├── 56_agent_custom.py             # 自定义代理
│   ├── 57_agent_custom.py             # 高级自定义代理
│   ├── 58_agent_searxng_simple.py     # SearxNG 简单搜索
│   ├── 59_agent_searxng_usage.py      # SearxNG 使用示例
│   ├── 60_simple_agent_search.py      # 简单搜索代理
│   ├── 61_agent_llm_search.py         # LLM 搜索代理
│   ├── 62_search_comparison.py        # 搜索方案对比
│   ├── 63_search_arXiv.py             # arXiv 学术搜索
│   └── data/                          # 代理数据
│       └── memory_data.json           # 记忆数据文件
├── chapter06/          # SalesGPT 智能销售代理系列
│   ├── 01_basic_salesGPT.py           # v1.0 基础版销售代理
│   ├── 02_enhanced_conversation_salesGPT.py  # v2.0 增强对话版
│   ├── 03_knowledge_based_salesGPT.py # v3.0 知识库版
│   ├── 04_rag_enhanced_salesGPT.py    # v4.0 RAG增强版
│   ├── 05_enterprise_salesGPT.py      # v5.0 企业版
│   ├── demo_all_versions.py           # 全版本演示脚本
│   ├── test_imports.py                # 导入测试脚本
│   ├── README.md                      # SalesGPT系列详细说明
│   └── data/                          # 数据文件
│       ├── car_knowledge_base.txt     # 产品知识库
│       ├── comprehensive_sales_data.json  # 综合销售数据
│       └── product_summary.json       # 产品摘要信息
└── .env                               # 环境变量配置文件
```

## 环境要求

- **Python**: 3.8+ (推荐 3.9+)
- **LangChain**: 最新版本 (0.1.0+)
- **模型服务**:
  - Ollama 本地服务 (推荐)
  - OpenAI API 或兼容服务
  - 智谱AI API (可选)
- **向量数据库**: FAISS, Chroma
- **其他依赖**: 详见各章节的 import 语句

### 主要依赖包

```bash
langchain
langchain-community
langchain-core
langchain-openai
langchain-ollama
langchain-chroma
faiss-cpu
python-dotenv
zhipuai
requests
nltk
pydantic
```

## 快速开始

### 1. 环境配置

创建 `.env` 文件并配置必要的 API 密钥：

```bash
# OpenAI API 配置 (必需)
OPENAI_API_KEY=your_openai_api_key_here

# 智谱AI API 配置 (可选)
ZHIPU_API_KEY=your_zhipu_api_key_here

# SerpAPI 搜索配置 (chapter05 需要)
SERPAPI_API_KEY=your_serpapi_key_here

# SearxNG 搜索服务配置 (可选)
SEARXNG_HOST=http://localhost:6688

# 天气API配置 (可选)
WEATHER_API_KEY=your_weather_api_key_here
```

### 2. 安装依赖

```bash
# 安装基础依赖
pip install langchain langchain-community langchain-core
pip install langchain-openai langchain-ollama langchain-chroma
pip install faiss-cpu python-dotenv requests nltk pydantic

# 安装特定服务依赖
pip install zhipuai  # 智谱AI
pip install google-search-results  # SerpAPI
```

### 3. 启动 Ollama 服务 (推荐)

```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 启动服务
ollama serve

# 下载推荐模型
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull quentinz/bge-large-zh-v1.5:latest
```

### 4. 运行示例

```bash
# 基础连接测试
python chapter00/06_connect_ollama_invoke.py

# 提示工程示例
python chapter01/08_similarity_chroma.py

# RAG 系统示例
python chapter03/37_rag_unstructured_pdfDirectoryLoader.py

# 智能代理示例
python chapter05/56_agent_custom.py

# SalesGPT 销售代理示例
python chapter06/01_basic_salesGPT.py
python chapter06/07_demo_all_versions.py
```

## 章节详解

### Chapter 00: 模型连接基础 🔌
学习如何连接和使用不同的大语言模型服务：
- **OpenAI API**: 标准调用和流式输出
- **智谱AI**: 国产模型服务集成
- **Ollama**: 本地模型部署和调用
- **自定义连接**: 适配其他模型服务

### Chapter 01: 提示工程与模板 📝
掌握提示词设计和模板使用技巧：
- **基础模板**: 聊天提示模板的创建和使用
- **少样本学习**: Few-shot 提示技术
- **示例选择器**: 智能选择最相关的示例
- **向量相似度**: 基于语义的示例匹配
- **模板组合**: 复杂提示模板的构建

### Chapter 02: 输出解析器 🔧
学习结构化输出处理和缓存机制：
- **格式解析**: CSV、JSON、XML、日期等格式
- **类型解析**: 枚举、自定义类型解析
- **缓存机制**: 内存和数据库缓存
- **错误处理**: 输出修复和容错机制

### Chapter 03: RAG 检索增强生成 🔍
构建强大的文档问答和检索系统：
- **向量检索**: FAISS、Chroma 向量数据库
- **文档加载**: 文本、PDF、HTML、Markdown 处理
- **文本分割**: 智能文档分割策略
- **检索优化**: 多种检索算法和优化技巧

### Chapter 04: Runnable 链式处理 ⛓️
掌握 LangChain 的核心处理链技术：
- **串行处理**: 顺序执行的处理链
- **并行处理**: 同时执行多个任务
- **数据传递**: 链间数据流转和处理
- **分支逻辑**: 条件分支和动态路由
- **自定义组件**: 创建自定义处理组件

### Chapter 05: 智能代理系统 🤖
构建具有工具调用能力的智能代理：
- **搜索代理**: 集成 SerpAPI、SearxNG 等搜索服务
- **工具集成**: 自定义工具和函数调用
- **记忆管理**: 对话历史和上下文管理
- **高级代理**: 多工具协作和复杂任务处理

### Chapter 06: SalesGPT 智能销售代理系列 🚀
从简单到复杂的渐进式销售AI开发：
- **v1.0 基础版**: 最简单的销售对话代理，学习基础概念
- **v2.0 增强对话版**: 智能阶段分析和改进的对话管理
- **v3.0 知识库版**: 集成关键词匹配的产品知识库
- **v4.0 RAG增强版**: 基于向量检索的智能知识问答
- **v5.0 企业版**: 完整的CRM和客户管理功能
- **演示系统**: 全版本对比演示和测试工具

## 核心功能特性

### 🔄 模型集成
- 支持 OpenAI、智谱AI、Ollama 等多种模型
- 统一的调用接口和错误处理
- 流式输出和批量处理
- 自定义模型适配

### 📊 文档处理
- 多格式支持：PDF、HTML、文本、Markdown
- 智能文本分割和向量化
- 批量文档加载和处理
- 文档元数据管理

### 🔍 检索系统
- 高效向量检索：FAISS、Chroma
- 语义相似度搜索
- 多向量检索策略
- 检索结果排序和过滤

### 🧠 智能代理
- 工具调用和函数执行
- 多步推理和规划
- 记忆管理和上下文保持
- 错误恢复和重试机制

### 🎯 输出控制
- 结构化输出解析
- 多种格式支持
- 输出验证和修复
- 自定义解析器

## 学习路径建议

### 🚀 初学者路径
1. **Chapter 00**: 从模型连接开始，了解基础调用
2. **Chapter 01**: 学习提示工程，掌握模板使用
3. **Chapter 02**: 理解输出解析，处理结构化数据
4. **Chapter 03**: 构建 RAG 系统，实现文档问答

### 🔥 进阶路径
1. **Chapter 04**: 掌握链式处理，构建复杂工作流
2. **Chapter 05**: 开发智能代理，集成外部工具
3. **Chapter 06**: SalesGPT系列，从简单到企业级的完整开发

### 💡 实践建议
- 每个示例都包含详细注释，建议逐行阅读理解
- 先运行基础示例，再尝试修改参数和配置
- 结合实际需求，选择合适的技术栈
- 关注错误处理和异常情况的处理方式

## 常见问题

### Q: 如何选择合适的模型？
- **轻量级任务**: 使用 qwen2.5:3b
- **复杂任务**: 使用 qwen2.5:7b 或 OpenAI GPT
- **中文优化**: 推荐使用 Qwen 系列模型

### Q: 向量模型如何选择？
- **中文文本**: quentinz/bge-large-zh-v1.5:latest
- **英文文本**: text-embedding-ada-002
- **多语言**: multilingual-e5-large

### Q: 如何优化检索效果？
- 合理设置文本分割大小 (chunk_size)
- 调整重叠长度 (chunk_overlap)
- 使用合适的相似度阈值
- 考虑使用重排序模型

### Q: SalesGPT系列应该从哪个版本开始学习？
- **初学者**: 从v1.0基础版开始，理解基本概念
- **有经验者**: 可以直接从v3.0知识库版开始
- **企业应用**: 重点学习v4.0和v5.0版本
- **快速体验**: 运行demo_all_versions.py查看所有版本对比

### Q: 如何自定义SalesGPT的产品信息？
- **v3.0版本**: 修改SimpleKnowledgeBase类中的knowledge_data
- **v4.0/v5.0版本**: 编辑data/car_knowledge_base.txt文件
- **销售流程**: 修改SALES_STAGES字典自定义销售阶段
- **人员信息**: 更新salesperson_info字典

## 贡献指南

欢迎提交 Issue 和 Pull Request！

### 贡献方式
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范
- 保持代码简洁易懂
- 添加详细的中文注释
- 遵循 Python PEP 8 规范
- 提供完整的示例和说明

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 强大的 LLM 应用开发框架
- [Ollama](https://ollama.ai/) - 优秀的本地模型运行平台
- [FAISS](https://github.com/facebookresearch/faiss) - 高效的向量检索库
- [Chroma](https://github.com/chroma-core/chroma) - 现代化的向量数据库

## 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交 GitHub Issue
- 发起 Discussion
- 贡献代码和文档

---

⭐ 如果这个项目对你有帮助，请给个 Star 支持一下！