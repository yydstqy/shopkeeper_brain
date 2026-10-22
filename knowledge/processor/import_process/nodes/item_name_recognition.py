import json
from typing import Tuple, List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):

    """
        商品名称识别
        利用3个chunk的内容上下文 小于2500字符，通过提示词模板让LLM识别出商品名称及型号，将商品名称信息bge-m3向量化，存储再milvus，供后续匹配使用
    """

    name = "item_name_recognition_node"

    def process(self,state : ImportGraphState):

        #1. 校验参数，获取file_title,chunks,item_name_chunk_k,item_name_chunk_size
        # config对象中属性不合法: ValidationError
        # state对象中属性不合法: StateFieldError
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = self.validate_state(state)

        #2. 构建识别商品名称上下文，取前三个切片，限制字符数
        item_name_recognition_context = self._prepare_context(chunks, item_name_chunk_k, item_name_chunk_size)

        #3. 通过提示词模板让LLM识别出商品名称及型号   AIClients   降级：file_title
        item_name = self._recognition_item_name(item_name_recognition_context, file_title)

        #4. 将商品名称信息bge-m3向量化
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        #5. 存储在milvus   StorageClients获取客户端，创建集合，数据保存
        #标量字段: pk, file_title, item_name
        #向量字段: dense_vector, sparse_vector
        self._insert_milvus(file_title, item_name, dense_vector, sparse_vector)

        #6.回填状态     将item_name回填到state和每一个chunk对象中
        self._fill_item_name_state(state, item_name,chunks)

        return state

    def validate_state(self, state) -> Tuple[str, List[Dict[str, Any]], int, int]:
        """
        校验参数，获取file_title,chunks,item_name_chunk_k,item_name_chunk_size
        :param state: ImportGraphState
            必须提供: file_title,chunks
        :return: file_title, chunks, item_name_chunk_k, item_name_chunk_size
        """

        file_title = state.get("file_title")
        chunks = state.get("chunks")
        item_name_chunk_k = self.config.item_name_chunk_k       #3
        item_name_chunk_size = self.config.item_name_chunk_size     #2500

        if not file_title:
            self.logger.error(f"file_title不存在")
            raise StateFieldError("file_title", expected_type=str)
        if not chunks or not isinstance(chunks, list):
            self.logger.error(f"chunks不存在")
            raise StateFieldError("chunks", expected_type=list)
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            self.logger.error(f"item_name_chunk_k不存在或小于等于零")
            raise ValidationError("item_name_chunk_k不存在或小于等于零", self.name)
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            self.logger.error(f"item_name_chunk_size不存在或小于等于零")
            raise ValidationError("item_name_chunk_size不存在或小于等于零", self.name)

        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    def _prepare_context(self, chunks, item_name_chunk_k, item_name_chunk_size):
        """
        构建识别商品名称上下文，取前三个切片，限制字符数
        :param chunks: 切片列表
        :param item_name_chunk_k: 切片数量
        :param item_name_chunk_size: 切片内容长度
        :return: 用于提取商品名称的上下文参考内容
        """

        chunks = chunks[0::item_name_chunk_k]
        final_context = []
        total = 0

        for index,chunk in enumerate(chunks):
            if not chunk and not isinstance(chunk,dict):
                continue

            chunk_content = chunk.get("content", "")
            content = f"【切片】-【{index+1}】\n" + chunk_content
            total += len(content)

            if len(content) < item_name_chunk_size:
                final_context.append(content)

        return "\n\n".join(final_context)

    def _recognition_item_name(self, item_name_recognition_context, file_title):
        """
        调用llm，获取客户端clients，识别商品名称
        :param item_name_recognition_context: 用于提取商品名称的上下文参考内容
        :param file_title: 文件标题
        :return: 商品名称
        """

        # 获取llm客户端
        llm_client = AIClients.get_llm_openai(False)       #返回文本不是json

        # 用户提示词
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_recognition_context)

        llm_response = llm_client.invoke([
            SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        if not llm_response or llm_response.content == "UNKNOWN":
            self.logger.error(f"LLM调用失败，降级处理，用文件名称作为商品名称")
            return file_title

        item_name = llm_response.content.strip()

        return item_name

    def _embedding_item_name(self, item_name) -> Tuple[List[float], Dict[int, float]]:
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error("BGE-M3模型客户端初始化失败", e)
            return None,None

        try:
            embeddings = bge_m3_client.encode_documents([item_name])

            dense_vector = embeddings["dense"][0].tolist()  # List[float], 长度 1024
            sparse_matrix = embeddings["sparse"]  # CSR 稀疏矩阵

            start_idx = sparse_matrix.indptr[0]
            end_idx = sparse_matrix.indptr[1]
            token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()
            weights = sparse_matrix.data[start_idx:end_idx].tolist()
            sparse_vector = dict(zip(token_ids, weights))  # Dict[int, float]

            return dense_vector,sparse_vector
        except Exception as e:
            self.logger.error("BGE-M3模型向量化商品名称失败", e)
            return None,None

    def _insert_milvus(self, file_title, item_name, dense_vector, sparse_vector):
        """
        将商品名称信息bge-m3向量化，存储在milvus
        :param file_title: 文件标题
        :param item_name: 商品名称
        :param dense_vector: 稠密向量
        :param sparse_vector: 稀疏向量
        :return:
        """

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error("Milvus客户端初始化失败", e)
            return

        item_name_collection = self.config.item_name_collection

        try:
            if not milvus_client.has_collection(item_name_collection):
                self.logger.info(f"集合 {item_name_collection} 不存在，正在创建集合...")
                self._create_item_name_collection(item_name_collection, milvus_client)
                self.logger.info(f"集合 {item_name_collection} 创建成功")
        except Exception as e:
            self.logger.error(f"集合 {item_name_collection} 创建失败", e)

        try:
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            self.logger.info(f"正在向集合 {item_name_collection} 插入数据...")
            milvus_client.insert(item_name_collection, data)
            self.logger.info(f"向集合 {item_name_collection} 插入数据成功")
        except Exception as e:
            self.logger.error(f"向集合 {item_name_collection} 插入数据失败", e)

    def _create_item_name_collection(self, collection_name, milvus_client):
        schema = milvus_client.create_schema()

        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)     #主键
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)      #标量
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)      #标量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)      #向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)      #向量

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector",
                              index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")     # 模糊近似查询  余弦相似度  只考虑方向
        index_param.add_index(field_name="sparse_vector",
                              index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")     # IP内积(查询更精准)  考虑方向和大小  归一化后和余弦相似

        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema, index_params=index_param
        )
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")

    def _fill_item_name_state(self, state, item_name, chunks):
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["item_name"] = item_name

if __name__ == '__main__':
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"D:\temp_dir\万用表RS-12的使用\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "万用表RS-12的使用_new",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    process_state = node(state)

    # 5. 输出结果
    print(f"商品名: {process_state.get('item_name')}")
    print(f"chunks数量: {len(process_state.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in process_state['chunks'][0]}")