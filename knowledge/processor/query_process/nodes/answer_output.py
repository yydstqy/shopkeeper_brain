from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState


class AnswerOutputNode(BaseNode):
    name = "answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        return state