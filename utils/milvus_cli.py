from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)


class MilvusClient:
    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
        self.is_connected = False

    def connect(self):
        if self.is_connected:
            try:
                utility.list_collections()
                print(f'👏 复用已连接的 Milvus at {self.host}:{self.port}')
                return
            except Exception as e:
                print(f"❌ 连接到 Milvus 失败: {e}")
                self.is_connected = False
        
        connections.connect(host=self.host, port=self.port)
        print(f"✅ 已连接到 Milvus at {self.host}:{self.port}")
        self.is_connected = True

    def get_collection(self, collection_name: str):
        self.connect()
        coll =  Collection(collection_name)
        return coll

def milvus_connection():
    cli = MilvusClient()
    cli.connect()

def create_milvus_collection(collection_name, dim=1024):
    # 检查集合是否存在
    if utility.has_collection(collection_name):
        print(f"⚠️  集合 '{collection_name}' 已存在")
        print(f"🗑️  自动删除并重新创建集合...")
        collection = Collection(collection_name)
        collection.drop()
        print(f"✅ 已删除集合 '{collection_name}'")
    
    # 定义字段
    fields = [
        FieldSchema(name="case_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ]
    
    schema = CollectionSchema(fields, description="Medical cases for hybrid retrieval")
    collection = Collection(collection_name, schema)
    
    # 创建索引（使用余弦相似度）
    index_params = {
        "metric_type": "COSINE",  # 余弦相似度
        "index_type": "FLAT",
        "params": {}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"✅ 创建集合 '{collection_name}' 成功（使用余弦相似度）")
    
    return collection


if __name__ == "__main__":
    cli = MilvusClient()
    cli.connect()
    coll = cli.get_collection("medical_cases")
    print(coll)
