import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from utils.es_cli import es_connection
from elasticsearch import helpers
from utils.milvus_cli import milvus_connection
from utils.embedding import to_embedding
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility


console = Console()

# ==================== 配置常量 ====================
ES_INDEX = "medical_2"
MILVUS_COLLECTION = "medical_2"


# ==================== 数据生成 ====================
def generate_medical_cases():
    medical_cases = [
        {
            "id": 1,
            "text": "李明，男，45岁，主诉胸痛3小时，既往高血压病史5年，血压160/95mmHg，心电图示ST段抬高，诊断急性心肌梗死，予以溶栓治疗后症状缓解。",
        },
        {
            "id": 2,
            "text": "王丽，女，32岁，主诉发热咳嗽5天，体温38.5℃，胸部CT示双肺多发磨玻璃影，核酸检测阳性，诊断新冠肺炎，给予抗病毒治疗。",
        },
        {
            "id": 3,
            "text": "张强，男，58岁，主诉腹痛腹胀2天，既往胆囊结石病史，腹部B超示胆囊壁增厚，血象白细胞15000/μL，诊断急性胆囊炎，建议手术治疗。",
        },
        {
            "id": 4,
            "text": "刘芳芳，女，28岁，主诉停经45天伴恶心呕吐，尿HCG阳性，B超示宫内早孕，孕囊大小符合孕6周，建议定期产检。",
        },
        {
            "id": 5,
            "text": "陈伟，男，65岁，主诉头晕头痛1月，血常规示血红蛋白85g/L，大便潜血阳性，胃镜检查发现胃角部溃疡，病理示腺癌，诊断胃癌。",
        },
        {
            "id": 6,
            "text": "赵敏，女，42岁，主诉右下腹痛6小时，体温37.8℃，McBurney点压痛阳性，血象白细胞12000/μL，诊断急性阑尾炎，予以抗感染治疗。",
        },
        {
            "id": 7,
            "text": "孙杰，男，35岁，主诉外伤后右腿疼痛，X线示右股骨中段骨折，无血管神经损伤，行切开复位内固定术，术后恢复良好。",
        },
        {
            "id": 8,
            "text": "周芳，女，50岁，主诉反复咳嗽咳痰2月，胸部CT示右肺上叶占位性病变，纤支镜活检示鳞状细胞癌，诊断肺癌，建议化疗。",
        },
        {
            "id": 9,
            "text": "吴涛，男，55岁，主诉腰痛伴下肢放射痛1周，腰椎MRI示L4-5椎间盘突出压迫神经根，保守治疗无效，建议手术治疗。",
        },
        {
            "id": 10,
            "text": "郑丽，女，38岁，主诉头痛头晕伴视物模糊3天，血压180/110mmHg，眼底检查示视网膜动脉硬化，诊断高血压危象，予以降压治疗。",
        },
        {
            "id": 11,
            "text": "黄建国，男，62岁，主诉胸闷气短1月余，超声心动图示左室射血分数35%，BNP升高，诊断慢性心力衰竭，予以强心利尿治疗。",
        },
        {
            "id": 12,
            "text": "林小红，女，29岁，主诉发热伴皮疹3天，体温39℃，全身散在红色斑丘疹，血常规示血小板减少，诊断登革热，予以对症支持治疗。",
        },
        {
            "id": 13,
            "text": "钱大伟，男，48岁，主诉多饮多尿多食伴体重下降2月，空腹血糖12.5mmol/L，餐后2小时血糖18.3mmol/L，糖化血红蛋白9.2%，诊断2型糖尿病。",
        },
        {
            "id": 14,
            "text": "孙梅，女，56岁，主诉双膝关节疼痛3年，X线示双膝关节间隙变窄、骨质增生，诊断骨关节炎，建议理疗及口服非甾体抗炎药。",
        },
        {
            "id": 15,
            "text": "朱晓明，男，41岁，主诉右侧腰部绞痛伴血尿4小时，泌尿系B超示右侧输尿管上段结石，直径0.6cm，予以解痉止痛及大量饮水排石治疗。",
        },
        {
            "id": 16,
            "text": "徐艳，女，33岁，主诉月经紊乱半年，B超示子宫肌瘤多发，最大者直径5cm，无明显症状，建议定期复查或考虑手术治疗。",
        },
        {
            "id": 17,
            "text": "何志强，男，59岁，主诉吞咽困难2月，胃镜示食管中段肿物，活检病理示鳞状细胞癌，诊断食管癌，建议手术或放化疗。",
        },
        {
            "id": 18,
            "text": "谢丽华，女，44岁，主诉甲状腺肿大1年，甲状腺B超示右叶结节，大小2.0×1.5cm，穿刺活检示乳头状癌，建议甲状腺全切术。",
        },
        {
            "id": 19,
            "text": "罗军，男，52岁，主诉右上腹痛伴黄疸1周，腹部CT示胰头占位，CA19-9明显升高，诊断胰腺癌，建议手术探查。",
        },
        {
            "id": 20,
            "text": "曾小燕，女，26岁，主诉右侧乳房肿块2月，乳腺B超示右乳2点钟方向实性结节，大小1.5cm，边界清楚，穿刺活检示纤维腺瘤，建议手术切除。",
        },
        {
            "id": 21,
            "text": "血糖如果很高的话？首先要控制饮食，减少糖分和碳水化合物摄入，多吃蔬菜和粗粮；其次要增加运动，每天至少30分钟有氧运动；定期监测血糖值；遵医嘱服用降糖药物如二甲双胍；严重者需注射胰岛素治疗；同时要控制体重，戒烟限酒，保持良好作息。建议定期复查糖化血红蛋白，预防并发症。",
        },
    ]

    return medical_cases


# ==================== Elasticsearch 操作 ====================
def create_es_index(client):
    """创建 Elasticsearch 索引"""
    # 检查索引是否存在
    if client.indices.exists(index=ES_INDEX):
        print(f"⚠️  索引 '{ES_INDEX}' 已存在")
        print(f"🗑️  自动删除并重新创建索引...")
        client.indices.delete(index=ES_INDEX)
        print(f"✅ 已删除索引 '{ES_INDEX}'")

    # 创建索引映射
    # 注意：如果安装了IK分词器，可以使用 "analyzer": "ik_max_word", "search_analyzer": "ik_smart"
    # 否则使用标准分词器
    mapping = {
        "mappings": {
            "properties": {
                "case_id": {"type": "integer"},
                "text": {
                    "type": "text",
                    "analyzer": "standard",  # 使用标准分词器（支持中文按字分词）
                    "search_analyzer": "standard",
                },
            }
        }
    }

    client.indices.create(index=ES_INDEX, body=mapping)
    print(f"✅ 创建索引 '{ES_INDEX}' 成功（使用标准分词器）")


def insert_to_es(client, cases):
    """将数据插入到 Elasticsearch"""
    print(f"\n📝 准备插入 {len(cases)} 条数据到 Elasticsearch...")

    # 准备批量操作
    bulk_actions = []
    for case in cases:
        action = {
            "_index": ES_INDEX,
            "_source": {"case_id": case["id"], "text": case["text"]},
        }
        bulk_actions.append(action)

    # 批量插入
    try:
        success_count, errors = helpers.bulk(
            client, bulk_actions, chunk_size=1000, request_timeout=60
        )
        print(f"✅ 成功插入 {success_count} 条数据到 Elasticsearch")

        # 刷新索引
        client.indices.refresh(index=ES_INDEX)

        # 检查文档数量
        count = client.count(index=ES_INDEX)
        print(f"📊 索引 '{ES_INDEX}' 当前文档总数: {count['count']}")

    except Exception as e:
        print(f"❌ 批量插入失败: {e}")
        raise


def search_es(client, query, top_k=5):
    """从 Elasticsearch 检索"""
    search_body = {"query": {"match": {"text": {"query": query}}}, "size": top_k}

    try:
        response = client.search(index=ES_INDEX, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append(
                {
                    "source": "ES",
                    "case_id": hit["_source"]["case_id"],
                    "text": hit["_source"]["text"],
                    "score": hit["_score"],
                }
            )

        return results

    except Exception as e:
        print(f"❌ ES 检索失败: {e}")
        return []


# ==================== Milvus 操作 ====================
def create_milvus_collection(dim=1024):
    """创建 Milvus 集合"""
    # 检查集合是否存在
    if utility.has_collection(MILVUS_COLLECTION):
        print(f"⚠️  集合 '{MILVUS_COLLECTION}' 已存在")
        print(f"🗑️  自动删除并重新创建集合...")
        collection = Collection(MILVUS_COLLECTION)
        collection.drop()
        print(f"✅ 已删除集合 '{MILVUS_COLLECTION}'")

    # 定义字段
    fields = [
        FieldSchema(name="case_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ]

    schema = CollectionSchema(fields, description="Medical cases for hybrid retrieval")
    collection = Collection(MILVUS_COLLECTION, schema)

    # 创建索引（使用余弦相似度）
    index_params = {
        "metric_type": "COSINE",  # 余弦相似度
        "index_type": "FLAT",
        "params": {},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"✅ 创建集合 '{MILVUS_COLLECTION}' 成功（使用余弦相似度）")

    return collection


def insert_to_milvus(collection, cases):
    """将数据插入到 Milvus（需要先向量化）"""
    print(f"\n📝 准备插入 {len(cases)} 条数据到 Milvus...")

    # 初始化 Embedding 处理器
    print("🔄 初始化 Embedding 处理器...")

    # 准备数据
    case_ids = []
    embeddings = []
    texts = []

    # 逐条处理并生成 embedding
    print("🔄 生成 Embeddings...")
    for case in tqdm(cases, desc="向量化"):
        embedding = to_embedding(case["text"])
        case_ids.append(case["id"])
        embeddings.append(embedding)
        texts.append(case["text"])

    # 准备插入数据
    insert_data = [case_ids, embeddings, texts]

    # 插入数据
    print("\n💾 插入数据到 Milvus...")
    try:
        collection.insert(insert_data)
        collection.flush()
        print(f"✅ 成功插入 {len(cases)} 条数据到 Milvus")

        # 加载集合
        collection.load()
        print(f"✅ 集合已加载，当前文档总数: {collection.num_entities}")

    except Exception as e:
        print(f"❌ 插入失败: {e}")
        raise


def search_milvus(collection, query, top_k=5):
    """从 Milvus 检索（使用余弦相似度）"""
    # 初始化 Embedding 处理器

    # 生成查询的 embedding
    query_embedding = to_embedding(query)

    # 搜索参数
    search_params = {"metric_type": "COSINE", "params": {}}  # 使用余弦相似度

    try:
        # 执行搜索
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["case_id", "text"],
        )

        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append(
                    {
                        "source": "Milvus",
                        "case_id": hit.entity.get("case_id"),
                        "text": hit.entity.get("text"),
                        "score": hit.distance,  # 余弦相似度分数
                    }
                )

        return formatted_results

    except Exception as e:
        print(f"❌ Milvus 检索失败: {e}")
        return []


# ==================== 混合检索 ====================
def hybrid_search(es_client, milvus_collection, query, top_k=5, alpha=0.5):
    """混合检索：从 ES 和 Milvus 分别检索，然后融合结果

    Args:
        es_client: Elasticsearch 客户端
        milvus_collection: Milvus 集合
        query: 查询文本
        top_k: 返回结果数量
        alpha: ES 和 Milvus 的权重（0-1），0表示全部使用Milvus，1表示全部使用ES

    Returns:
        (融合后的结果列表, ES结果, Milvus结果)
    """
    print(f"🔍 查询: {query}")
    print("=" * 80)

    # 从 ES 检索
    print("\n1️⃣  从 Elasticsearch 检索...")
    es_results = search_es(es_client, query, top_k=top_k)
    print(f"   找到 {len(es_results)} 条结果")

    # 从 Milvus 检索
    print("\n2️⃣  从 Milvus 检索...")
    milvus_results = search_milvus(milvus_collection, query, top_k=top_k)
    print(f"   找到 {len(milvus_results)} 条结果")

    # 结果融合
    print("\n3️⃣  融合检索结果...")
    merged_results = merge_results(es_results, milvus_results, alpha)

    # 排序并返回 top_k
    merged_results.sort(key=lambda x: x["final_score"], reverse=True)
    return merged_results[:top_k], es_results, milvus_results


def merge_results(es_results, milvus_results, alpha=0.5):
    """融合 ES 和 Milvus 的检索结果

    融合策略：
    - 对于同一个 case_id，计算加权分数：final_score = alpha * es_score + (1-alpha) * milvus_score
    - 对于只在一个源中出现的结果，使用该源的分数
    """

    # 归一化分数到 [0, 1]
    def normalize_scores(results):
        if not results:
            return results

        max_score = max(r["score"] for r in results)
        min_score = min(r["score"] for r in results)

        if max_score == min_score:
            for r in results:
                r["normalized_score"] = 1.0
        else:
            for r in results:
                r["normalized_score"] = (r["score"] - min_score) / (
                    max_score - min_score
                )

        return results

    # 归一化分数
    es_results = normalize_scores(es_results)
    milvus_results = normalize_scores(milvus_results)

    # 构建结果字典
    result_dict = {}

    # 添加 ES 结果
    for result in es_results:
        case_id = result["case_id"]
        result_dict[case_id] = {
            "case_id": case_id,
            "text": result["text"],
            "es_score": result["normalized_score"],
            "milvus_score": 0.0,
            "sources": ["ES"],
        }

    # 添加 Milvus 结果
    for result in milvus_results:
        case_id = result["case_id"]
        if case_id in result_dict:
            result_dict[case_id]["milvus_score"] = result["normalized_score"]
            result_dict[case_id]["sources"].append("Milvus")
        else:
            result_dict[case_id] = {
                "case_id": case_id,
                "text": result["text"],
                "es_score": 0.0,
                "milvus_score": result["normalized_score"],
                "sources": ["Milvus"],
            }

    # 计算最终分数
    merged_results = []
    for case_id, data in result_dict.items():
        final_score = alpha * data["es_score"] + (1 - alpha) * data["milvus_score"]
        merged_results.append(
            {
                "case_id": data["case_id"],
                "text": data["text"],
                "es_score": data["es_score"],
                "milvus_score": data["milvus_score"],
                "final_score": final_score,
                "sources": ", ".join(data["sources"]),
            }
        )

    return merged_results


def display_results(results, es_results=None, milvus_results=None):
    """显示检索结果"""
    # 1. 显示ES检索结果
    if es_results:
        print("\n" + "=" * 80)
        print("📊 Elasticsearch 检索结果（关键词匹配）")
        print("=" * 80)

        if not es_results:
            print("❌ ES 没有找到相关结果")
        else:
            es_table = []
            for i, result in enumerate(es_results, 1):
                es_table.append(
                    [
                        i,
                        result["case_id"],
                        (
                            result["text"][:80] + "..."
                            if len(result["text"]) > 80
                            else result["text"]
                        ),
                        f"{result['score']:.4f}",
                    ]
                )

            # 创建Rich表格
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("排名", justify="center", style="cyan", width=6)
            table.add_column("案例ID", justify="center", style="green", width=8)
            table.add_column("文本", style="white")
            table.add_column("分数", justify="right", style="yellow", width=12)

            for row in es_table:
                table.add_row(*[str(x) for x in row])

            console.print(table)

            # 显示完整文本
            console.print("\n[bold green]💡 完整文本：[/bold green]")
            for i, result in enumerate(es_results, 1):
                console.print(
                    f"\n[bold]{i}. [案例 {result['case_id']}][/bold] [yellow](分数: {result['score']:.4f})[/yellow]"
                )
                console.print(f"   {result['text']}")

    # 2. 显示Milvus检索结果
    if milvus_results:
        print("\n" + "=" * 80)
        print("📊 Milvus 检索结果（语义相似度）")
        print("=" * 80)

        if not milvus_results:
            print("❌ Milvus 没有找到相关结果")
        else:
            milvus_table = []
            for i, result in enumerate(milvus_results, 1):
                milvus_table.append(
                    [
                        i,
                        result["case_id"],
                        (
                            result["text"][:80] + "..."
                            if len(result["text"]) > 80
                            else result["text"]
                        ),
                        f"{result['score']:.4f}",
                    ]
                )

            # 创建Rich表格
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("排名", justify="center", style="cyan", width=6)
            table.add_column("案例ID", justify="center", style="green", width=8)
            table.add_column("文本", style="white")
            table.add_column("分数", justify="right", style="yellow", width=12)

            for row in milvus_table:
                table.add_row(*[str(x) for x in row])

            console.print(table)

            # 显示完整文本
            console.print("\n[bold green]💡 完整文本：[/bold green]")
            for i, result in enumerate(milvus_results, 1):
                console.print(
                    f"\n[bold]{i}. [案例 {result['case_id']}][/bold] [yellow](分数: {result['score']:.4f})[/yellow]"
                )
                console.print(f"   {result['text']}")

    # 3. 显示融合后的结果
    if not results:
        print("\n❌ 融合后没有找到相关结果")
        return

    print("\n" + "=" * 80)
    print("🔥 融合检索结果（ES + Milvus 加权融合）")
    print("=" * 80)

    # 准备表格数据
    table_data = []
    for i, result in enumerate(results, 1):
        table_data.append(
            [
                i,
                result["case_id"],
                (
                    result["text"][:60] + "..."
                    if len(result["text"]) > 60
                    else result["text"]
                ),
                f"{result['es_score']:.4f}",
                f"{result['milvus_score']:.4f}",
                f"{result['final_score']:.4f}",
                result["sources"],
            ]
        )

    # 创建Rich表格
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("排名", justify="center", style="cyan", width=6)
    table.add_column("案例ID", justify="center", style="green", width=8)
    table.add_column("文本", style="white", width=50)
    table.add_column("ES分数", justify="right", style="blue", width=10)
    table.add_column("Milvus分数", justify="right", style="blue", width=12)
    table.add_column("最终分数", justify="right", style="red", width=10)
    table.add_column("来源", justify="center", style="green", width=15)

    for row in table_data:
        table.add_row(*[str(x) for x in row])

    console.print(table)

    # 显示完整文本
    console.print("\n[bold green]💡 完整文本：[/bold green]")
    for i, result in enumerate(results, 1):
        console.print(
            f"\n[bold]{i}. [案例 {result['case_id']}][/bold] [red](最终分数: {result['final_score']:.4f})[/red]"
        )
        console.print(f"   {result['text']}")


def setup_data():
    """初始化数据：创建索引/集合并插入数据（只需要运行一次）"""
    print("=" * 80)
    print("🏥 医疗案例混合检索系统 - 数据初始化")
    print("=" * 80)

    # 生成测试数据
    medical_cases = generate_medical_cases()
    print(f"\n📊 已生成 {len(medical_cases)} 条医疗案例数据")

    print("\n" + "=" * 80)
    print("🔧 初始化数据")
    print("=" * 80)

    # 连接 Elasticsearch
    print("\n🔗 连接到 Elasticsearch...")
    es_client = es_connection()
    print("✅ Elasticsearch 连接成功")

    # 创建 ES 索引并插入数据
    create_es_index(es_client)
    insert_to_es(es_client, medical_cases)

    # 连接 Milvus
    print("\n🔗 连接到 Milvus...")
    milvus_connection()
    print("✅ Milvus 连接成功")

    # 创建 Milvus 集合并插入数据
    milvus_coll = create_milvus_collection(dim=1024)
    insert_to_milvus(milvus_coll, medical_cases)

    print("\n" + "=" * 80)
    print("🎉 数据初始化完成！")
    print("=" * 80)


def demo_search():
    print("\n" + "=" * 80)
    print("Elasticsearch、Milvus 混合检索")
    print("=" * 80)

    # 连接 Elasticsearch
    print("\n🔗 连接到 Elasticsearch...")
    es_client = es_connection()
    print("✅ Elasticsearch 连接成功")

    # 连接 Milvus
    print("\n🔗 连接到 Milvus...")
    milvus_connection()
    print("✅ Milvus 连接成功")

    milvus_coll = Collection(MILVUS_COLLECTION)
    milvus_coll.load()

    # 查询列表
    # 格式：("查询文本", top_k, alpha)
    # alpha: 0.0-1.0, 越大越依赖ES关键词匹配，越小越依赖Milvus语义搜索
    test_queries = [
        ("胸痛心肌梗死", 5, 0.5),  # 心血管疾病（平衡）
        ("肿瘤癌症治疗", 5, 0.5),  # 癌症相关（平衡）
        ("骨折外伤", 3, 0.5),  # 骨科问题（平衡）
        ("糖尿病", 3, 0.8),  # 关键词搜索（偏ES）
        ("血糖高怎么办", 3, 0.2),  # 语义搜索（偏Milvus）
    ]

    #  执行所有测试查询
    for i, (query, top_k, alpha) in enumerate(test_queries, 1):
        print("\n" + "=" * 80)
        print(f"📝 测试查询 {i}/{len(test_queries)}")
        print("=" * 80)

        # 执行检索
        results, es_results, milvus_results = hybrid_search(
            es_client, milvus_coll, query, top_k=top_k, alpha=alpha
        )

        # 显示结果（包括ES、Milvus、融合结果）
        display_results(results, es_results, milvus_results)

        if i < len(test_queries):
            print('\n准备下一个查询...\n')

    print("\n" + "=" * 80)
    print("🎉 所有检索测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    # setup_data()
    demo_search()
