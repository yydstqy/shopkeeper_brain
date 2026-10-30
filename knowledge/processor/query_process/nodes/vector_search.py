import json
from typing import Dict, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, item_names_filter, execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    name = "search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState|Dict:
        """
        三路检索之一: 向量检索
        :param state: rewritten_query, item_names
        :return:
            {
                "embedding_chunks": []
            }
        """

        #1. 参数校验
        validated_rewritten_query, validated_item_names = self._validate_input(state)

        #2. 获取客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error("Failed to get milvus client", e)
            return {}

        try:
            embedding_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error("Failed to get embedding model", e)
            return {}
        try:
            #3. 查询向量化
            embedding_hybrid_result = generate_bge_m3_hybrid_vectors(embedding_model, [validated_rewritten_query])

            #4. 构建查询条件
            expr, expr_params = item_names_filter(validated_item_names)       # 标量查询条件

            #5. 创建混合检索请求
            search_requests = create_hybrid_search_requests(        # 混合查询条件
                dense_vector=embedding_hybrid_result["dense"][0],
                sparse_vector=embedding_hybrid_result["sparse"][0],
                expr=expr,
                expr_params=expr_params
            )

            #6. 执行混合检索
            hybrid_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=search_requests,
                output_fields=["chunk_id", "item_name", "content", "title"]
            )

            #7. 回填数据
            return {
                "embedding_chunks": hybrid_result[0]
            }
        except Exception as e:
            self.logger.error("Failed to execute vector search", e)
            return {}

    def _validate_input(self, state) -> Tuple[str, list]:
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error("Invalid rewritten_query")
            raise StateFieldError(self.name, "rewritten_query", str)

        item_names = state.get("item_names")
        if not item_names or not isinstance(item_names, list):
            self.logger.error("Invalid item_names")
            raise StateFieldError(self.name, "item_names", list)

        return rewritten_query,item_names

if __name__ == '__main__':
    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"]
    }

    vector_search = VectorSearchNode()
    result = vector_search(state)

    for r in result.get('embedding_chunks', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))