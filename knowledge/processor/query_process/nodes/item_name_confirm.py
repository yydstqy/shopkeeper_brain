import json
import re
from json import JSONDecodeError
from typing import List, Dict, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from markdown_it.presets import default
from openai.types import create_embedding_response
from pymilvus import AnnSearchRequest
from pymilvus.client import search_result
from sympy import true
from sympy.matrices import dense

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.config import QueryConfig
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages


class ItemNameExtractor:
    """LLM商品名称识别; 清洗LLM返回结果"""

    def __init__(self, logger, node_name):
        self.logger = logger
        self.name = node_name

    def extract_item_name(self, original_query, history_text) -> Dict[str, Any]:
        """
        调用 LLM 获取商品名称
        :param original_query: 原始问题
        :param history_text: 历史会话 (10条消息，也即 5 轮对话)
        :return:
            {
            "item_names": ["万用表", "电阻"]  # 商品名称列表
            "rewritten_query": "万用表怎么测量电阻"
            }
        """

        result = {"item_names": [], "rewritten_query": original_query}  # 默认结果

        llm_client = AIClients.get_llm_openai(True)

        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(query=original_query, history_text=history_text)

        llm_response:AIMessage = llm_client.invoke([
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        llm_content = llm_response.content.strip()

        # 清洗返回结果
        try:
            clean_llm_content_json = self._clean_parse(llm_content)

            result["item_names"] = clean_llm_content_json.get("item_names", [])
            result["rewritten_query"] = clean_llm_content_json.get("rewritten_query", original_query)
            return result
        except Exception as e:
            self.logger.error(f"LLM返回结果清洗失败：{str(e)}")
            return result

    def _clean_parse(self, llm_content: str) -> Dict[str, Any]:
        """清洗并解析 LLM 响应"""
        # 1. 清洗 json 代码块围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 2. 反序列化
        try:
            parsed_llm_result: Dict[str, Any] = json.loads(content)
            # 2.1 清洗 item_names
            rwa_item_names = parsed_llm_result.get('item_names')
            if not isinstance(rwa_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [raw_item for raw_item in rwa_item_names if raw_item.strip()]

            # 2.2 清洗 rewritten_query
            raw_rewritten_query = parsed_llm_result.get('rewritten_query')
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

            return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")


class ItemNameAligner:
    """向量化查询，商品名称对齐评分，分数差过滤"""

    def __init__(self, logger, node_name, config:QueryConfig):
        self.logger = logger
        self.name = node_name
        self.config = config

    def match_align_filter(self, item_names) -> Tuple[List[str], List[str]]:
        """
        # 3.向量化匹配
        # 4.评分对齐
        # 5.分数差异过滤(仅当 confirmed 长度大于 1 )
        """

        """  extracted_name就是LLM根据用户问题识别出的商品名
            search_results: List[Dict[str, Any]]
            [
                {
                    "extracted_name":extracted_name,
                    "matches":[
                        {"item_name":"商品A1","score":0.9}
                        {"item_name":"商品A2","score":0.8}
                    ]   
                }
                {
                    "extracted_name":extracted_name,
                    "matches":[
                        {"item_name":"商品B1","score":0.7}
                        {"item_name":"商品B2","score":0.6}
                    ]   
                }
            ]
        """

        # 3.向量化匹配
        search_results:List[Dict[str, Any]] = self._match_vector(item_names)

        # 4.评分对齐
        # confirmed: 存放 得分>0.7 的商品名称
        # options: 存放 0.6<得分<0.7 的商品名称

        confirmed, options = self._item_name_score_align(search_results)

        # 5.分数差异过滤(仅当 confirmed 长度大于 1 )
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed,search_results)

        return confirmed, options

    def _match_vector(self, item_names):
        """
        向量匹配: 根据 LLM 识别商品名称, 从 Milvus 中查询相似商品名称(hybrid 混合向量检索)
        :param item_names: LLM识别商品名称列表
        :return: search_results
        """

        search_results: List[Dict[str, Any]] = []

        #1.获取milvus客户端
        milvus_client = StorageClients.get_milvus_client()
        if not milvus_client:
            return search_results

        #2.获取向量转换模型     BGE-M3
        embedding_client = AIClients.get_bge_m3_client()
        if not embedding_client:
            return search_results

        #3.调用工具类批量转换商品名称为向量
        # item_name_embedding =
        #{
        #   "dense":[[],[]],
        #   "sparse": [{},{}]
        # }
        item_name_embedding = generate_bge_m3_hybrid_vectors(embedding_client, item_names)

        #4.循环查询milvus中商品名称,混合检索
        for index,extract_item_name in enumerate(item_names):
            dense = item_name_embedding["dense"][index]
            sparse = item_name_embedding["sparse"][index]

            #4.1构建查询对象
            hybrid_search_requests:List[AnnSearchRequest] = create_hybrid_search_requests(dense, sparse)

            #4.2执行混合检索
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client = milvus_client,
                collection_name = self.config.item_name_collection,
                search_requests = hybrid_search_requests,
                output_fields=["item_name"]
            )

            #4.3封装数据
            search_result = {
                "extracted_name": extract_item_name,
                "matches": [
                    {"item_name":h["entity"]["item_name"], "score": h["distance"]}
                    for h in hybrid_search_result[0] if hybrid_search_result
                ]
            }
            search_results.append(search_result)

        return search_results

    def _item_name_score_align(self, search_results) -> Tuple[List[str], List[str]]:
        """
        评分对齐: 根据向量匹配结果, 对商品名称进行评分对齐
        规则:
            1.score >= 0.7  高分区     不超过 3 个
            2.0.7 > score >= 0.6  中分区   不超过 3 个
            3.score < 0.6  低分区
        """
        """
        search_results: List[Dict[str, Any]]
        [
            {
                "extracted_name": extracted_name,
                "matches": [
                    {"item_name": "商品A1", "score": 0.9}
                    {"item_name": "商品A2", "score": 0.8}
                ]
            }
            {
                "extracted_name": extracted_name,
                "matches": [
                    {"item_name": "商品B1", "score": 0.7}
                    {"item_name": "商品B2", "score": 0.6}
                ]
            }
        ]
        """

        confirmed = []
        options = []
        for search_result in search_results:
            extracted_name = search_result["extracted_name"]
            matches = sorted(search_result["matches"], key=lambda x: x["score"], reverse=True)

            high = [match["item_name"] for match in matches if match["score"] >= 0.7]
            if high:
                # 找到同名的则退出, 不再添加;没找到就返回None
                samename = next((item_name for item_name in high if item_name == extracted_name), None)
                if samename:
                    if samename not in confirmed:
                        confirmed.append(samename)
                elif len(high) == 1:
                    if high[0] not in confirmed:
                        confirmed.append(high[0])
                else:
                    for h in high[:3]:
                        if h not in options and h not in confirmed:
                            options.append(h)
            else:
                mid = [match["item_name"] for match in matches
                    if 0.6 <= match["score"] < 0.7
                    and match["item_name"] not in options
                    and match["item_name"] not in confirmed]
                if mid:
                    for m in mid[:3]:
                        if m not in options and m not in confirmed:
                            options.append(m)

        return confirmed, options[:3]

    def _item_name_score_filter(self, confirmed, search_results):
        """
        分数差异过滤: 对于 confirmed 中的商品名称, 进行分数差异过滤
        :param confirmed:
        :return:
        """

        high_confirmed_item_score = {}   # 高置信的商品名称和分数的字典

        for search_result in search_results:
            extracted_name = search_result["extracted_name"]
            matches = search_result["matches"]
            for match in matches:
                item_name = match["item_name"]
                score = match["score"]
                if item_name in confirmed:
                    high_confirmed_item_score[item_name] = max(high_confirmed_item_score.get(item_name, 0), score)

        sorted_high_confirmed_item_score = sorted(high_confirmed_item_score.items(), key=lambda x: x[1], reverse=True)
        high_score = sorted_high_confirmed_item_score[0][1]     # 最高分

        # 过滤掉分数差异大于0.15的商品名称
        confirmed = [item_name for item_name, score in high_confirmed_item_score.items() if high_score - score <= 0.15]
        return confirmed

class ItemNameConfirmNode(BaseNode):
    name = "item_name_confirm"

    def __init__(self):
        super().__init__()
        self._item_name_extractor = ItemNameExtractor(self.logger,self.name)
        self._item_name_aligner = ItemNameAligner(self.logger, self.name,self.config)

    def process(self, state: QueryGraphState) -> QueryGraphState:

        """
        负责从用户提问中提取并确认商品名称
            1.获取历史会话
            {
                "_id": {
                "$oid": "6a48cc0f82601667381db7a5"
                },
                "session_id": "sess-f41icdzoh87mr5zeel1",
                "role": "user",
                "text": "万用表怎么测量电阻",
                "rewritten_query": "万用表怎么测量电阻",
                "item_names": [],
                "ts": 1783155727.498414
            }
            {
              "_id": {
                "$oid": "6a48cc0f82601667381db7a6"
              },
              "session_id": "sess-f41icdzoh87mr5zeel1",
              "role": "assistant",
              "text": "我不确定您指的是哪款产品。, 您是在询问以下产品吗: RS PRO RS-12 数字万用表、HAK 180 扫描仪？",
              "rewritten_query": "万用表怎么测量电阻",
              "item_names": [],
              "ts": 1783155727.52313
            }
            2.LLM获取商品名称
            3.向量化匹配
            4.评分对齐
            5.分数差异过滤(仅当 confirmed 长度大于 1 )
            6.决策更新状态
            7.历史回填(仅当 confirmed 有值)
        """

        session_id = state.get("session_id")
        original_query = state.get("original_query")

        # 1.获取历史会话
        history_messages: List[Dict[str, Any]] = get_recent_messages(session_id)

        history_text = ""
        for message in history_messages:
            history_text += f"{message['role']}: {message['text']}\n"

        # 2.LLM获取商品名称
        extract_item_names:Dict[str, Any] = self._item_name_extractor.extract_item_name(original_query, history_text)
        # print(extract_item_names)
        item_names = extract_item_names.get("item_names")
        rewritten_query = extract_item_names.get("rewritten_query")

        # 3.向量化匹配
        # 4.评分对齐
        # 5.分数差异过滤(仅当 confirmed 长度大于 1 )
        # confirmed: 存放 得分>0.7 的商品名称
        # options: 存放 0.6<得分<0.7 的商品名称
        # state['answer'] 存在 -> 选项或者抱歉
        # state['answer'] 不存在 -> 商品名称已确定
        if item_names:
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []

        # 6.决策更新状态
        self._decide(state, confirmed, options,rewritten_query)

        # 7.历史回填(仅当 confirmed 有值)
        for history in history_messages:
            history["item_names"] = confirmed
        state["history"] = history_messages

        return state

    def _decide(self, state, confirmed, options, rewritten_query):
        """
        如果 confirmed 存在,直接确认,进行三路检索
        如果 options 存在,生成 state["answer"] 提示用户选择
        否则生成 state["answer"] 抱歉
        """

        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = f"我不确定您指的是哪款商品,您是在询问以下商品吗:{"、".join(options)}?"
        else:
            state["answer"] = "抱歉, 未找到相关商品名称,请提供更准确的产品名称或型号"


if __name__ == '__main__':
    item_name_confirm_node = ItemNameConfirmNode()
    init_state = {
        "session_id": "sess-f41icdzoh87mr5zeel1",
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        # "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        "original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
    }
    process_state = item_name_confirm_node(init_state)

    print(json.dumps(process_state,indent=4,ensure_ascii=False,default=str))