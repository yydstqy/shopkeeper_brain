import json
from pathlib import Path


from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.query_process.exceptions import ValidationError


class EntryNode(BaseNode):

    """
        入口节点:根据上传文件后缀来决定走哪个分支节点
    """

    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        self.log_step("step1", "获取上传文件以及路径")
        #1.获取上传文件以及路径
        import_file_path = state.get("import_file_path","")
        file_dir = state.get("file_dir","")

        self.log_step("step2", "判断文件以及路径是否存在")
        #2.判断文件以及路径是否为空
        if not import_file_path:
            raise ValidationError(f"上传文件不能为空",self.name)
        if not file_dir:
            raise ValidationError(f"上传文件路径不能为空",self.name)

        #3.获取文件后缀并变为小写
        path = Path(import_file_path)
        suffix = path.suffix.lower()

        self.log_step("step3", "判断文件类型")
        #4.不同的文件类型做不同的状态修改
        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise ValidationError(f"不支持的文件类型：{suffix}",self.name)

        #5.获取文件标题(不带扩展名)
        file_name = path.stem

        #6.更新并返回状态
        state["file_title"] = file_name
        return state

if __name__ == "__main__":

    setup_logging()

    import_file_path = r"D:\doc\万用表RS-12的使用.pdf"
    file_dir = r"D:\temp_dir"

    state = {
        "is_md_read_enabled": False,
        "is_pdf_read_enabled": False,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }
    entry = EntryNode()
    process_state = entry(state)
    print(json.dumps(process_state,indent=4,ensure_ascii=False))