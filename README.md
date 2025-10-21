# RAG 混合检索系统

基于 Elasticsearch 和 Milvus 的混合检索系统，结合关键词匹配和语义相似度检索，提供更准确的搜索结果。

## 📋 项目简介

本项目实现了一个完整的混合检索（Hybrid Search）系统，融合了：

- **Elasticsearch**：基于 BM25 算法的关键词匹配检索
- **Milvus**：基于向量的语义相似度检索
- **加权融合**：可调节的检索结果融合策略

## ✨ 功能特性

- 🔍 **双模检索**：同时支持关键词匹配和语义搜索
- ⚖️ **灵活融合**：可调节 alpha 参数控制两种检索方式的权重
- 📊 **可视化结果**：使用 Rich 库美化输出，清晰展示检索结果
- 🚀 **高性能**：Elasticsearch 提供快速关键词匹配，Milvus 提供高效向量检索
- 🎯 **实时向量化**：使用 Ollama bge-m3 模型进行文本向量化

## 🛠️ 技术栈

- **Python 3.10+**
- **Elasticsearch 8.x**：全文检索引擎
- **Milvus 2.x**：向量数据库
- **Ollama**：本地部署的 embedding 模型（bge-m3）
- **Rich**：命令行输出美化
- **tqdm**：进度条显示

## 📦 安装

### 1. 环境要求

- Python 3.10 或更高版本
- Elasticsearch（运行在 localhost:9200）
- Milvus（运行在 localhost:19530）
- Ollama（安装 bge-m3 模型）

### 2. 安装依赖

```bash
pip install elasticsearch elasticsearch-dsl
pip install pymilvus
pip install ollama
pip install rich tqdm
```

### 3. 启动服务

**Elasticsearch**

```bash
# Docker 方式
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=true" \
  -e "ELASTIC_PASSWORD=your_password" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

**Milvus**

```bash
# Docker 方式
docker run -d --name milvus \
  -p 19530:19530 \
  milvusdb/milvus:latest
```

**Ollama + bge-m3**

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取 bge-m3 模型
ollama pull bge-m3
```

## 🚀 快速开始

### 1. 配置连接信息

编辑 `utils/es_cli.py` 和 `utils/milvus_cli.py`，配置你的连接信息。

### 2. 修改索引名称

编辑 `hybrid/es_mvs.py`，修改常量：

```python
ES_INDEX = "your_index_name"
MILVUS_COLLECTION = "your_collection_name"
```

### 3. 初始化数据

第一次运行时，需要初始化数据：

```python
# 在 es_mvs.py 中
if __name__ == "__main__":
    setup_data()    # 初始化数据（仅第一次运行）
    demo_search()   # 执行检索演示
```

```bash
python hybrid/es_mvs.py
```

### 4. 执行检索

初始化完成后，注释掉 `setup_data()`：

```python
if __name__ == "__main__":
    # setup_data()  # 已初始化，注释掉
    demo_search()
```

再次运行即可测试检索功能。

## 📂 项目结构

```
RAG/
├── hybrid/
│   └── es_mvs.py          # 混合检索主程序
├── utils/
│   ├── __init__.py
│   ├── es_cli.py          # Elasticsearch 连接工具
│   ├── milvus_cli.py      # Milvus 连接工具
│   └── embedding.py       # 文本向量化工具
├── .gitignore
└── README.md
```

## 🔧 配置说明

### Alpha 参数

控制 Elasticsearch 和 Milvus 的权重：

- `alpha = 1.0`：完全依赖 Elasticsearch（关键词匹配）
- `alpha = 0.5`：均衡使用两种检索方式
- `alpha = 0.0`：完全依赖 Milvus（语义相似度）

### 修改查询

在 `demo_search()` 函数中修改测试查询：

```python
test_queries = [
    ("你的查询", 5, 0.5),  # (查询文本, 返回数量, alpha值)
    ("另一个查询", 3, 0.8),
]
```

## 📊 使用示例

### 示例输出

```
================================================================================
Elasticsearch、Milvus 混合检索
================================================================================

🔗 连接到 Elasticsearch...
✅ Elasticsearch 连接成功

🔗 连接到 Milvus...
✅ Milvus 连接成功

================================================================================
📝 测试查询 1/3
================================================================================
🔍 查询: 胸痛心肌梗死
================================================================================

1️⃣  从 Elasticsearch 检索...
   找到 5 条结果

2️⃣  从 Milvus 检索...
   找到 5 条结果

3️⃣  融合检索结果...

================================================================================
🔥 Hybrid Search Results (ES + Milvus Weighted Fusion)
================================================================================
┏━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Rank ┃  ID   ┃ Text                       ┃ ES Score ┃ Milvus Score ┃ Final Score ┃   Source   ┃
┡━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│  1   │   1   │ Patient with chest pain... │  1.0000  │    1.0000    │   1.0000    │ ES, Milvus │
│  2   │  11   │ Patient with heart fail... │  0.2770  │    0.3622    │   0.3196    │ ES, Milvus │
│  3   │   5   │ Patient with gastric ca... │  0.0000  │    0.1684    │   0.0842    │   Milvus   │
└──────┴───────┴────────────────────────────┴──────────┴──────────────┴─────────────┴────────────┘
```

## 🎯 核心功能

### 1. 数据初始化

```python
setup_data()  # 创建索引、集合并插入数据
```

### 2. 混合检索

```python
results, es_results, milvus_results = hybrid_search(
    es_client=es_client,
    milvus_collection=milvus_collection,
    query="你的查询",
    top_k=5,
    alpha=0.5
)
```

### 3. 结果展示

```python
display_results(results, es_results, milvus_results)
```

## 🔍 工作原理

1. **关键词检索**：Elasticsearch 使用 BM25 算法进行关键词匹配
2. **语义检索**：将查询文本向量化，在 Milvus 中进行余弦相似度搜索
3. **分数归一化**：将两种检索的分数归一化到 [0, 1] 区间
4. **加权融合**：`final_score = alpha * es_score + (1-alpha) * milvus_score`
5. **结果排序**：按融合分数降序返回 top-k 结果

## ⚠️ 注意事项

1. **首次运行**：必须先执行 `setup_data()` 初始化数据
2. **重复初始化**：每次运行 `setup_data()` 会删除旧数据并重新创建
3. **向量化耗时**：首次插入数据时需要生成 embedding，会比较慢
4. **模型选择**：默认使用 `bge-m3` 模型，可在 `utils/embedding.py` 中修改

## 📝 自定义数据

修改 `generate_medical_cases()` 函数添加你的数据：

```python
def generate_medical_cases():
    medical_cases = [
        {
            "id": 1,
            "text": "你的文本内容",
        },
        # 添加更多数据...
    ]
    return medical_cases
```

## 🐛 常见问题

**Q: 连接 Elasticsearch/Milvus 失败？**  
A: 检查服务是否正常运行，端口是否正确，认证信息是否配置。

**Q: 向量化速度慢？**  
A: bge-m3 模型较大，首次加载会慢一些。可以考虑使用更小的模型或 GPU 加速。

**Q: 搜索结果不准确？**  
A: 调整 alpha 参数，或者增加训练数据量。

## 📄 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请提交 Issue。
