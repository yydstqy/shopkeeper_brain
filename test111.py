from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):

    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        #1 获取上传文件以及路径
        import_file_path = state.get("import_file_path","")
        file_dir = state.get("file_dir","")

        #2 判断上传文件及路径是否为空
        if not import_file_path:
            raise ValidationError("上传文件为空")
        if not file_dir:
            raise ValidationError("上传文件路径为空")

        #3 获取文件后缀并改为小写
        path = Path(import_file_path)
        suffix = path.suffix.lower()

        #4 不同的文件类型做不同的修改
        if suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        elif suffix == ".pdf"
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        else:
            raise ValidationError("不支持的文件类型")

        #5 获取文件标题
        state["file_title"] = path.stem

        return state