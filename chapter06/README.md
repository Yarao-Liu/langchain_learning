# SalesGPT 渐进式开发系列

这是一个从简单到复杂的SalesGPT智能销售代理系统开发系列，展示了如何逐步构建一个功能完整的企业级销售AI助手。

## 📚 版本概览

### v1.0 - 基础版 (`01_basic_salesGPT.py`)
**最简单的销售对话代理**

**功能特点：**
- ✅ 基础的对话阶段管理
- ✅ 简单的销售流程
- ✅ 基本的用户交互
- ✅ 轻量级实现

**适用场景：**
- 学习LangChain基础概念
- 理解销售对话流程
- 快速原型验证

**运行方式：**
```bash
python 01_basic_salesGPT.py
```

---

### v2.0 - 增强对话版 (`02_enhanced_conversation_salesGPT.py`)
**改进的对话管理系统**

**功能特点：**
- ✅ 智能的阶段分析系统
- ✅ 更丰富的销售人员信息
- ✅ 改进的对话管理
- ✅ 更好的错误处理
- ✅ 对话状态跟踪

**新增功能：**
- 使用LLM进行阶段分析
- 更详细的销售流程
- 改进的提示词工程
- 对话上下文管理

**运行方式：**
```bash
python 02_enhanced_conversation_salesGPT.py
```

---

### v3.0 - 知识库版 (`03_knowledge_based_salesGPT.py`)
**集成简单知识库系统**

**功能特点：**
- ✅ 基于关键词的知识库系统
- ✅ 产品信息管理
- ✅ 智能信息检索
- ✅ 上下文感知回复
- ✅ 结构化产品数据

**新增功能：**
- 无需复杂依赖的知识库
- 快速关键词匹配
- 丰富的产品信息
- 智能信息推荐

**运行方式：**
```bash
python 03_knowledge_based_salesGPT.py
```

---

### v4.0 - RAG增强版 (`04_rag_enhanced_salesGPT.py`)
**集成向量检索系统**

**功能特点：**
- ✅ 向量嵌入和检索系统
- ✅ RetrievalQA集成
- ✅ 智能文档检索
- ✅ 高级知识问答
- ✅ 多种嵌入模型支持

**新增功能：**
- 基于向量的语义检索
- RetrievalQA问答系统
- 多种嵌入模型备选
- 智能文档分割

**依赖要求：**
```bash
pip install sentence-transformers
pip install faiss-cpu
```

**运行方式：**
```bash
python 04_rag_enhanced_salesGPT.py
```

---

### v5.0 - 企业版 (`05_enterprise_salesGPT.py`)
**完整的企业级销售代理系统**

**功能特点：**
- ✅ 客户档案管理系统
- ✅ 销售数据分析
- ✅ 多渠道集成支持
- ✅ 完整的CRM功能
- ✅ 销售流程自动化
- ✅ 性能监控和报告

**系统架构：**
- **EnterpriseSalesGPT**: 核心销售代理，统一协调所有组件
- **CustomerManager**: 客户管理系统，处理客户档案和互动记录
- **SalesAnalytics**: 销售数据分析系统，提供洞察和建议
- **EnterpriseKnowledgeBase**: 企业知识库，采用RAG架构实现智能检索

**典型工作流程：**
1. **客户咨询进入** → EnterpriseSalesGPT接收用户输入
2. **客户识别** → CustomerManager查找客户档案信息
3. **知识检索** → EnterpriseKnowledgeBase搜索相关产品信息
4. **阶段判断** → 阶段分析链确定当前销售阶段
5. **个性化回复** → 对话链生成专业回复
6. **数据记录** → CustomerManager记录本次互动
7. **分析洞察** → SalesAnalytics更新客户分析

**新增功能：**
- 完整的客户生命周期管理
- 智能销售预测和分析
- 多渠道统一管理（电话、邮件、视频会议等）
- 自动化销售流程
- 实时性能监控和报告
- RAG架构的企业知识库
- 客户参与度评分算法
- 智能行动建议系统

**运行方式：**
```bash
python 05_enterprise_salesGPT.py
```

## 🏗️ 系统架构分析

### 企业版 SalesGPT v5.0 架构概览

企业版采用**多层次、多组件协作的架构模式**，实现了真正的企业级销售代理系统：

#### 🧠 核心代理层
- **EnterpriseSalesGPT**: 主代理系统，作为整个系统的指挥中心
  - 统筹管理所有业务组件
  - 协调客户管理、数据分析和知识库系统
  - 控制对话流程和销售阶段推进

#### 🏢 业务组件层
- **CustomerManager**: 客户管理系统
  - 客户档案存储和管理
  - 销售互动历史记录
  - 客户状态跟踪

- **SalesAnalytics**: 销售数据分析系统
  - 销售表现分析
  - 客户参与度评分
  - 智能行动建议

- **EnterpriseKnowledgeBase**: 企业知识库系统
  - RAG架构实现语义检索
  - FAISS向量数据库
  - RetrievalQA智能问答

#### 🔗 LangChain技术栈
- **对话生成链**: 个性化销售对话
- **阶段分析链**: 智能销售阶段判断
- **知识检索链**: 产品信息检索问答
- **向量嵌入**: OllamaEmbeddings语义理解

#### 📊 数据基础层
- **CustomerProfile**: 客户档案数据结构
- **SalesInteraction**: 销售互动记录
- **枚举类**: 标准化状态和渠道定义

### 调用关系特点

1. **Agent vs Chain**: EnterpriseSalesGPT是真正的Agent，具有状态管理和决策能力；其他组件主要提供专业化功能支持

2. **RAG集成**: 知识库系统采用检索增强生成架构，实现智能的产品信息检索

3. **数据驱动**: 所有决策基于客户数据和互动历史，实现个性化服务

4. **模块化设计**: 各组件职责明确，便于独立开发和维护

## 🚀 快速开始

### 环境准备

1. **安装依赖**
```bash
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install python-dotenv
pip install faiss-cpu  # 用于v4.0和v5.0
pip install sentence-transformers  # 用于v4.0和v5.0
```

2. **配置环境变量**
创建 `.env` 文件：
```env
OPENAI_API_KEY=your_api_key_here
```

3. **选择版本运行**
根据需求选择合适的版本运行。

## 📊 版本对比

| 功能特性 | v1.0 | v2.0 | v3.0 | v4.0 | v5.0 |
|---------|------|------|------|------|------|
| 基础对话 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 智能阶段分析 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 关键词知识库 | ❌ | ❌ | ✅ | ❌ | ❌ |
| 向量检索(RAG) | ❌ | ❌ | ❌ | ✅ | ✅ |
| 客户档案管理 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 销售数据分析 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 多渠道支持 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 企业知识库 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 客户洞察分析 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 自动化流程 | ❌ | ❌ | ❌ | ❌ | ✅ |
| Agent架构 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 复杂度 | 低 | 中 | 中 | 高 | 很高 |
| 适用场景 | 学习 | 学习 | 演示 | 产品 | 生产 |

## 🎯 学习路径建议

### 初学者路径
1. **v1.0 基础版** - 理解基本概念
2. **v2.0 增强版** - 学习提示词工程
3. **v3.0 知识库版** - 了解信息检索

### 进阶路径
1. **v4.0 RAG增强版** - 掌握向量检索
2. **v5.0 企业版** - 学习系统架构

### 实际应用路径
- **快速原型**: 使用 v1.0 或 v2.0
- **产品演示**: 使用 v3.0 或 v4.0
- **生产部署**: 使用 v5.0

## 🔧 自定义指南

### 修改销售人员信息
在每个版本中找到 `salesperson_info` 字典，修改相应字段：
```python
self.salesperson_info = {
    "name": "你的名字",
    "role": "你的角色",
    "company": "你的公司",
    # ... 其他信息
}
```

### 添加产品信息
- **v3.0**: 修改 `SimpleKnowledgeBase` 类中的 `knowledge_data`
- **v4.0/v5.0**: 修改知识库文件内容或调用 `_create_enterprise_knowledge_file()` 方法

### 自定义销售阶段
修改 `SALES_STAGES` 字典：
```python
SALES_STAGES = {
    "1": "你的阶段1描述",
    "2": "你的阶段2描述",
    # ... 更多阶段
}
```

### 企业版特殊配置 (v5.0)

#### 配置嵌入模型
企业版优先使用本地Ollama嵌入模型：
```bash
# 安装并启动Ollama
ollama pull quentinz/bge-large-zh-v1.5:latest
```

#### 自定义客户数据
修改 `CustomerManager._create_sample_customers()` 方法：
```python
def _create_sample_customers(self):
    sample_customers = [
        CustomerProfile(
            customer_id="YOUR_CUST_ID",
            name="客户姓名",
            company="客户公司",
            # ... 更多客户信息
        )
    ]
```

#### 配置分析算法
在 `SalesAnalytics` 类中自定义评分算法：
```python
def _calculate_engagement_score(self, interactions):
    # 自定义参与度评分逻辑
    pass
```

## 📁 文件结构

```
chapter06/64_salesGPT/
├── 01_basic_salesGPT.py           # v1.0 基础版
├── 02_enhanced_conversation_salesGPT.py  # v2.0 增强对话版
├── 03_knowledge_based_salesGPT.py # v3.0 知识库版
├── 04_rag_enhanced_salesGPT.py    # v4.0 RAG增强版
├── 05_enterprise_salesGPT.py      # v5.0 企业版
├── README.md                      # 本文件
├── 64_agent_salesGPT.py          # 原始版本
├── 65_enhanced_salesGPT_with_RAG.py  # 原始增强版
└── 66_simple_salesGPT_with_knowledge.py  # 原始简化版

chapter06/data/
├── car_knowledge_base.txt         # v4.0 RAG知识库文件
├── enterprise_knowledge_base.txt  # v5.0 企业知识库文件（自动生成）
├── customers.json                 # v5.0 客户数据（运行时生成）
├── interactions.json              # v5.0 交互记录（运行时生成）
└── comprehensive_sales_data.json  # 综合销售数据
```

## 🤝 贡献指南

欢迎提交改进建议和bug报告！

## 📄 许可证

本项目仅用于学习和研究目的。

---

**开始你的SalesGPT开发之旅吧！** 🚀
