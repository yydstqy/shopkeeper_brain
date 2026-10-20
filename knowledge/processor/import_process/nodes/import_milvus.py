from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.state import ImportGraphState


class ImportMilvusNode(BaseNode):

    name = "import_milvus_node"

    def process(self, state: ImportGraphState):
        return state