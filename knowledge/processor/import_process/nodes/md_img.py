import base64
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Tuple, List, Dict, Deque
from threading import Lock

from langchain_openai import OpenAI

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    """图片在 MD 中的上下文信息。"""
    heading: str      # 最近的章节标题
    pre_text: str     # 图片上方的正文内容
    post_text: str    # 图片下方的正文内容


@dataclass
class ImageInfo:
    """一张图片的完整信息。"""
    name: str                # 图片文件名，如 "abc123.jpg"
    path: str                # 图片完整路径
    context: ImageContext    # 在 MD 中的上下文


"""读取&备份"""
class MdFileHandler:

    def __init__(self,logger:Logger, node_name:str):
        self.logger = logger
        self.node_name = node_name

    def read_md(self,state:ImportGraphState) -> Tuple[str,Path,Path]:

        self.logger.info("读取md文件内容，返回md内容，md路径，图片目录")

        md_path = state.get("md_path","")
        if not md_path:
            raise StateFieldError(node_name=self.node_name, field_name="md_path", expected_type=str)

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError("md_path文件不存在")

        with open(md_path_obj, 'r', encoding='utf-8') as f:
            md_content = f.read()

        images_dir = md_path_obj.parent / "images"

        return md_content,md_path_obj,images_dir

    def backup(self, new_md_content, md_path_obj):
        self.logger.info("备份md文件")
        #  markdown.md -> 万用表RS-12的使用_new.md
        backup_path = md_path_obj.parent / (md_path_obj.parent.name + "_new.md")
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(new_md_content)


"""扫描图片上下文"""
class ImageScanner:

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def scan_img_dir(self, md_content: str, images_dir: Path, image_extensions: set, context_length: int) -> List[ImageInfo]:

        """
        扫描每一张图片的上下文信息，返回一个图片信息列表
        :param md_content: md文档内容
        :param images_dir: 图片目录     D:\\temp_dir\\万用表RS-12的使用\\images
        :param image_extensions: 图片扩展名集合{jpg, png, gif,jpeg,bmp,webp}
        :param context_length: 上下文最大长度
        :return: 图片信息列表
        """

        image_info_list: List[ImageInfo] = []

        for image_path in images_dir.iterdir():

            if not image_path.is_file():        #不是文件直接跳过
                continue

            if not image_path.suffix in image_extensions:        #不是图片格式直接跳过
                # raise ValidationError(f"图片{image_path} - 格式错误：{image_path.suffix}")
                continue

            # 查找图片上下文，如果是表格图片在md中可能找不到，返回None
            ctx = self._find_context(md_content,image_path.name,context_length)
            if ctx is None:
                continue

            image_info_list.append(ImageInfo(name=image_path.name,path=image_path,context=ctx))

        return  image_info_list

    def _find_context(self, md_content: str, img_name: str, max_chars: int = 200) -> ImageContext | None:

        """返回图片在 MD 中第一次出现位置的上下文，找不到返回 None。"""
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)"
        )
        #将md文档拆成一行一行的字符串元素集合
        md_lines = md_content.split("\n")

        for line_idx, line in enumerate(md_lines):

            if not pattern.search(line):
                continue

            # 向上：找最近标题，取标题到图片之间的内容作为上文
            prev_title, prev_boundary = self._find_heading_above(md_lines, line_idx)
            pre_content = md_lines[prev_boundary + 1: line_idx]
            img_pre = self._extract_limited_context(
                pre_content, max_chars, direction="front"
            )

            # 向下：找下一个标题，取图片到标题之间的内容作为下文
            next_boundary = self._find_heading_below(md_lines, line_idx)
            post_content = md_lines[line_idx + 1: next_boundary]
            img_post = self._extract_limited_context(
                post_content, max_chars, direction="end"
            )

            return ImageContext(
                heading=prev_title,
                pre_text=img_pre,
                post_text=img_post,
            )

        return None

    @staticmethod
    def _find_heading_above(
        md_lines: List[str], from_idx: int
    ) -> Tuple[str, int]:
        """从 from_idx 向上查找最近的标题。"""
        for i in range(from_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i
        return "", -1

    @staticmethod
    def _find_heading_below(md_lines: List[str], from_idx: int) -> int:
        """从 from_idx 向下查找下一个标题。"""
        for i in range(from_idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i
        return len(md_lines)

    @staticmethod
    def _extract_limited_context(
        lines: List[str], max_chars: int, direction: str
    ) -> str:
        """按段落分割，按 direction 方向贪心装填，保持段落完整性。"""
        current_paragraph: List[str] = []
        paragraphs: List[str] = []

        for line in lines:
            #line.strip(): 去除字符串首尾的空白字符(空格、制表符、换行符等)
            is_blank_line = not line.strip()
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )

            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue

            current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        if direction == "front":
            paragraphs.reverse()#就近原则

        total = 0
        selected: List[str] = []
        for para in paragraphs:
            if (total + len(para) > max_chars) and selected:#至少有个段落
                break
            selected.append(para)
            total += len(para)

        if direction == "front":
            selected.reverse()#与原文顺序一致，利于VLM

        return "\n\n".join(selected)#折行并空一行


"""VLM摘要"""
class VLMSummarizer:

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name
        self.stampts_deque:Deque[float] = deque()
        self.lock:Lock = Lock()

    def summarize_all(self, document_name:str, imageinfo_list:List[ImageInfo], vl_model:str, requests_per_minute=5) -> Dict[str, str]:

        summaries: Dict[str, str] = {}
        # stamps_deque : Deque[float] = deque()     不能每次请求都创建一个队列

        try:
            openai_client = AIClients.get_openai()
        except Exception as e:
            self.logger.info(f"获取OpenAI客户端失败{e}")
            for image_info in imageinfo_list:
                summaries[image_info.name] = "默认图片摘要"
                return summaries

        for image_info in imageinfo_list:
            #限流
            self._enforce_rate_limit(requests_per_minute)

            #调用vlm生成摘要
            summary = self._summarize_one(image_info,openai_client,vl_model,document_name)
            summaries[image_info.name] = summary

        return summaries

    def _summarize_one(self, image_info:ImageInfo, openai_client:OpenAI, vl_model:str, document_name:str) -> str:

        """
        :param image_info:图片上下文信息
        :param openai_client:vlm模型客户端对象     单例
        :param vl_model:vlm模型名称
        :param document_name:文档名称
        :return:
        """

        # 图片转 Base64
        with open(image_info.path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        parts = [p for p in (image_info.context.heading,image_info.context.pre_text,image_info.context.post_text) if p]
        final_context = "\n".join(parts)

        # 调用 VLM
        response = openai_client.chat.completions.create(
            model=vl_model,  # 视觉模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",  # 告诉 API：这是一段文字
                            "text":(
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",  # 告诉 API：这是一张图片
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )

        summary = response.choices[0].message.content
        return summary

    def _enforce_rate_limit(self,  requests_per_minute,window:int=60):

        """
        限流
        :param requests_per_minute: 每分钟请求次数限制
        :return:
        """

        with self.lock:
            now = time.time()

            while self.stampts_deque and now - self.stampts_deque[0] > window:
                self.stampts_deque.popleft()

            if len(self.stampts_deque) >= requests_per_minute:
                sleep_duration = window - (now - self.stampts_deque[0])
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                now = time.time()
                while self.stampts_deque and now - self.stampts_deque[0] > window:
                    self.stampts_deque.popleft()

            self.stampts_deque.append(now)


"""图片存储minio & 替换"""
class ImageUploader:

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def upload_and_replace(self, document_name:str, md_content:str, imageinfo_list:List[ImageInfo], summaries:Dict[str,str], minio_bucket:str,
                           minio_endpoint:str) -> str:
        """
        上传图片到minio，替换路径和摘要
        :param document_name: 不带扩展名的文件名字
        :param md_content: md文档内容
        :param imageinfo_list: 图片信息列表
        :param summaries: 图片摘要列表
        :param minio_bucket: 图片上传到指定的桶
        :param minio_endpoint: minio地址
        :return: 新的md
        """

        #图片名称对应图片url
        remote_urls:Dict[str,str] = self._upload_all(imageinfo_list,document_name,minio_bucket,minio_endpoint)

        new_md_content = self._replace_in_md(remote_urls=remote_urls,md_content=md_content,summaries=summaries)

        return new_md_content

    def _upload_all(self, imageinfo_list, document_name, minio_bucket, minio_endpoint):
        """
        上传图片到minio
        :param imageinfo_list: 图片信息列表
        :param document_name: md文件父目录名称
        :param minio_bucket: MinIO 桶名称
        :param minio_endpoint: MinIO 端点
        :return: Dict{图片名称:对应图片url}
        """

        remote_urls:Dict[str,str] = {}

        try:
            minio_client = StorageClients.get_minio_client()
        except Exception:
            #主降级处理，当获取不到连接时，所有图片降级处理
            for imageinfo in imageinfo_list:
                remote_urls[imageinfo.name] = imageinfo.path
            return remote_urls

        for image_info in imageinfo_list:
            try:
                minio_client.fput_object(bucket_name=minio_bucket,
                                         object_name=f"{document_name}/{image_info.name}",
                                         file_path=image_info.path)
                remote_url = f"{minio_endpoint}/{minio_bucket}/{document_name}/{image_info.name}"
                remote_urls[image_info.name] = remote_url

            except Exception:
                # 二级降级处理，只针对上传失败图片进行降级处理
                remote_urls[image_info.name] = image_info.path

        return remote_urls

    def _replace_in_md(self, remote_urls:Dict[str,str], md_content:str, summaries:Dict[str,str]):
        """
        替换摘要和路径，返回新的md
        :param remote_urls: 图片名称:对应图片url
        :param md_content: md文档内容
        :param summaries: 图片摘要列表
        :return: 新的md
        """
        pattern = re.compile(r"!\[(.*?)]\((.*?)\)")

        def replacer(match: re.Match) -> str:
            original_url = match.group(2).strip()
            img_url_md = Path(original_url).name
            for img_name, summary in summaries.items():
                if img_name == img_url_md:
                    return f"![{summary}]({remote_urls[img_name]})"
            return match.group(0)


        return pattern.sub(replacer,md_content)



class MarkDownImageNode(BaseNode):

    name = "md_img_node"

    def __init__(self):
        super().__init__()
        self.md_file_handler = MdFileHandler(self.logger,self.name)
        self.image_scanner = ImageScanner(self.logger,self.name)
        self.vlm_summarizer = VLMSummarizer(self.logger,self.name)
        self.image_uploader = ImageUploader(self.logger,self.name)

    def process(self,state : ImportGraphState):

        #1. 文件处理：获取整个md文档内容，md路径,图片路径
        md_content,md_path_obj,images_dir = self.md_file_handler.read_md(state)

        if  not images_dir.exists():
            state["md_content"] = md_content
            return  state

        document_name = md_path_obj.parent.name

        #2. 获取上下文
        imageinfo_list:List[ImageInfo] = self.image_scanner.scan_img_dir(md_content, images_dir,
                                                                         image_extensions = self.config.image_extensions,
                                                                         context_length = self.config.img_content_length)

        #3. vlm生成图片摘要
        summaries:Dict[str,str] = self.vlm_summarizer.summarize_all(document_name=document_name,
                                                                     imageinfo_list=imageinfo_list,
                                                                     vl_model=self.config.vl_model,
                                                                     requests_per_minute=self.config.requests_per_minute)
        print(summaries)

        #4. 上传图片到minio，替换路径和摘要
        new_md_content:str = self.image_uploader.upload_and_replace(document_name=document_name,
                                               md_content=md_content,
                                               imageinfo_list=imageinfo_list,
                                               summaries=summaries,
                                               minio_bucket=self.config.minio_bucket,
                                               minio_endpoint=self.config.get_minio_base_url())

        #5. 备份
        self.md_file_handler.backup(new_md_content,md_path_obj)

        return state

if __name__ == "__main__":
    setup_logging()

    node = MarkDownImageNode()
    state = {
        "md_path": r"D:\temp_dir\万用表RS-12的使用\markdown.md"
    }
    process_state = node(state)
    print(json.dumps(process_state,indent = 4,ensure_ascii=False))