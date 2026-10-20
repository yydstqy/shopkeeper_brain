import json
import subprocess
import zipfile
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.query_process.exceptions import ValidationError


class PdfToMdNode(BaseNode):

    """
        使用MinerU工具将PDF文件转换为Markdown文件

        #   mineru-kit parse D:\\doc\\万用表RS-12的使用.pdf -o D:\temp_dir
        #   mineru-kit parse D:\\doc\\万用表RS-12的使用.pdf -o D:\temp_dir --tier standard
    """

    name = "pdf_to_node"

    def process(self,state : ImportGraphState):

        """

        :param state:
            state = {
        "is_md_read_enabled": False,
        "is_pdf_read_enabled": True,
        "import_file_path": D:\\doc\\万用表RS-12的使用.pdf,
        "file_dir": D:\\temp_dir
    }
        :return:

        """
        #1.参数校验
        import_file_path_obj, file_dir_path_obj = self._validate_state_inputs_path(state)

        #2.用mineru命令转换PDF为Markdown
        processed_code = self._execute_mineru(import_file_path_obj, file_dir_path_obj)
        if processed_code == 0:
            self.logger.info(f"PDF转Md成功!{import_file_path_obj} - 输出目录:{file_dir_path_obj}")
        else:
            self.logger.error(f"PDF转Md失败!")
            raise PdfConversionError("PDF转Md失败", self.name)

        #3.解压
        extract_dir = self._unzip_result(file_dir_path_obj, import_file_path_obj)

        #4.获取md文件路径
        md_path = self._get_md_paths(import_file_path_obj, extract_dir)

        #5.修改并返回状态
        state["md_path"] = md_path

        return state

    def _validate_state_inputs_path(self,state:ImportGraphState) -> Tuple[Path, Path]:

        """校验输入参数：文件路径，输出目录"""
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")

        if not import_file_path:        #防"调用方没给值"(state里到底有没有内容)
            raise ValidationError("输入文件路径为空", self.name)

        import_file_path_obj = Path(import_file_path)

        if not import_file_path_obj.exists():       #防"给了值但文件不在"(磁盘里到底有没有内容)
            raise FileProcessingError("输入文件路径不存在", self.name)

        if not file_dir:
            file_dir = import_file_path_obj.parent      #如果输出路径不存在就用输入文件的父目录作为输出路径

        file_dir_path_obj = Path(file_dir)

        return import_file_path_obj, file_dir_path_obj

    def _execute_mineru(self, import_file_path_obj, file_dir_path_obj) -> int:

        proc = subprocess.Popen(
            args=["mineru-kit", "parse", import_file_path_obj,
                  "-o", file_dir_path_obj, "--tier", "standard", "--format", "zip"],
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.STDOUT,  # 合并错误到标准输出
            text=True,
            encoding="utf-8",
            errors="replace",  # 遇到乱码时替换
            bufsize=1  # 行缓冲，实时输出
        )

        for line in proc.stdout:
            print(line.rstrip())

        processed_code = proc.wait()

        print(processed_code)

        return processed_code

    def _unzip_result(self, file_dir_path_obj, import_file_path_obj) -> Path:
        """解压 {stem}.zip"""
        zip_path = file_dir_path_obj / (import_file_path_obj.stem + ".zip")
        if not zip_path.exists():
            raise PdfConversionError("未找到 mineru-kit 输出的 zip", self.name)

        extract_dir = file_dir_path_obj / import_file_path_obj.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        zip_path.unlink()

        return extract_dir

    def _get_md_paths(self, import_file_path_obj, extract_dir) -> str:
        """获取md文件路径"""
        return str(extract_dir/"markdown.md")




if __name__ == "__main__":

    setup_logging()

    import_file_path = r"D:\doc\万用表RS-12的使用.pdf"
    file_dir = r"D:\temp_dir"

    state = {
        "is_md_read_enabled": True,
        "is_pdf_read_enabled": False,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    pdfToMdNode = PdfToMdNode()
    process_state = pdfToMdNode(state)
    print(json.dumps(process_state,indent=4,ensure_ascii=False))