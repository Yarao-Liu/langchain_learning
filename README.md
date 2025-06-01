# LangChain 中文实践项目

这是一个基于 LangChain 框架的中文实践项目，展示了如何使用 LangChain 构建各种 AI 应用，包括文本处理、文档问答、向量检索等功能。

## 项目特点

- 使用最新的 LangChain 框架
- 支持中文处理和问答
- 集成了多种文档加载器（PDF、文本等）
- 使用 FAISS 向量数据库进行高效检索
- 基于 Ollama 的本地模型部署
- 包含详细的代码注释和说明

## 目录结构

```
.
├── chapter01/          # 基础概念和入门示例
│   ├── 01.py          # 基础配置和模型调用
│   ├── 02.py          # 提示模板使用
│   └── ...
├── chapter02/          # 输出解析器实践
│   ├── 21.py          # 基础输出解析
│   ├── 22.py          # 结构化输出解析
│   └── ...
├── chapter03/          # RAG（检索增强生成）实践
│   ├── 34_rag_unstructured_textloader.py    # 文本文件处理
│   ├── 35_rag_unstructured_DirectoryLoader.py  # 目录文件处理
│   └── 37_rag_unstructured_pdfDirectoryLoader.py  # PDF文件处理
└── text/              # 示例文档目录
    ├── 01.txt        # 示例文本文件
    └── 01.pdf        # 示例PDF文件
```

## 环境要求

- Python 3.8+
- LangChain 最新版本
- Ollama 本地服务
- 其他依赖包（见 requirements.txt）

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动 Ollama 服务：
```bash
ollama serve
```

3. 运行示例：
```bash
python chapter03/37_rag_unstructured_pdfDirectoryLoader.py
```

## 主要功能

### 1. 文本处理
- 支持多种文本格式的加载和处理
- 文本分割和向量化
- 相似度搜索和检索

### 2. PDF 处理
- PDF 文档加载和解析
- 页面分割和向量化
- 基于内容的问答

### 3. 目录处理
- 批量文件加载
- 递归目录搜索
- 多文件内容检索

### 4. 问答系统
- 基于上下文的问答
- 相似度阈值过滤
- 结构化输出解析