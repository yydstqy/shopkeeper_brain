import asyncio
from typing import Dict, Tuple

from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMcpNode(BaseNode):
    name = "web_search_mcp"

    def process(self, state: QueryGraphState) -> Dict:
        """
        三路检索之一： 基于web联网的MCP服务调用。
        :param state: rewritten_query,item_names
        :return:
            {
                "web_search_docs": []
            }
        """

        # 1.参数校验
        validateed_rewritten_query, validated_item_names = self._validate_input(state)

        # 2.创建MCP客户端
        # 异步函数调用放在事件循环中，变成同步处理
        web_search_docs = asyncio.run(self._web_mcp(validateed_rewritten_query))

        # 5.封装返回结果
        return {
            "web_search_docs": web_search_docs
        }

    def _validate_input(self, state) -> Tuple[str, list]:
        # 1.参数校验
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error(f"Invalid rewritten_query: {rewritten_query}")
            raise StateFieldError(self.name, "rewritten_query", str)

        item_names = state.get("item_names")
        if not item_names or not isinstance(item_names, list):
            self.logger.error(f"Invalid item_names: {item_names}")
            raise StateFieldError(self.name, "item_names", str)
        return rewritten_query, item_names

    async def _web_mcp(self, validateed_rewritten_query):
        async with MCPServerStreamableHttp(
                name="网络搜索",
                params={
                    "url": self.config.mcp_dashscope_base_url,  # MCP 服务端点
                    "headers": {"Authorization": f"Bearer {self.config.openai_api_key}"},  # 认证头
                    "timeout": 300,  # 请求超时时间（秒）
                    "terminate_on_close": True,  # 关闭时终止连接
                },
                max_retry_attempts=2,  # 最大重试次数
                cache_tools_list=True,  # 缓存工具列表，避免重复请求
        ) as client:

            # 3.调用 call_tool    bailian_web_search
            execute_tool_result = await client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": validateed_rewritten_query, "count": 3}
            )
            print(execute_tool_result)

            # 4.解析结果
            if not execute_tool_result or not execute_tool_result.content or not execute_tool_result.content[0]:
                return []
            text_json = json.loads(execute_tool_result.content[0].text)
            if not text_json:
                return []
            pages = text_json.get("pages")
            if not pages:
                return []
            web_search_docs = []
            for page in pages:
                snippet = page.get("snippet")
                title = page.get("title")
                url = page.get("url")
                web_search_docs.append(
                    {
                        "snippet": snippet,
                        "title": title,
                        "url": url
                    }
                )
            return web_search_docs


if __name__ == '__main__':
    import json

    state = {
        "rewritten_query": "今天的小米汽车的股价是多少",
        "item_names": ["RS-12 数字万用表"]
    }

    web_mcp_search = WebSearchMcpNode()
    result = web_mcp_search(state)

    for r in result.get('web_search_docs', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))