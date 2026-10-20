from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState


class ItemNameRecognitionNode(BaseNode):

    name = "item_name_recognition_node"

    def process(self,state : ImportGraphState):
        return state