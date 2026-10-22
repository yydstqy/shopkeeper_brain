import json
import os
import re
from typing import Any, List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):

    name = "document_spilt_node"

    def process(self,state: ImportGraphState):

        """
        文档切分处理:
            1.获取输入参数并校验:md_content,file_title(无标题段落的默认标题),max_content_length,min_content_length

            2.按照标题切分: 按\n获取文档行的集合，遍历，判断是否存在代码块，正则表达式识别标题，构建section
                section = [{
                    "title": "string",  # 标题
                    "parent_title": "string", # 父标题
                    "file_title": "string", # 文件名称
                    "body": "string",  # 内容             content指"标题+body"
                },
                {
                    "title": "string",  # 标题
                    "parent_title": "string", # 父标题
                    "file_title": "string", # 文件名称
                    "body": "string",  # 内容
                }
                ]

            3.合并或拆分

            4.组装chunk:
                chunk = [{
                    "title": "string",  # 标题
                    "parent_title": "string", # 父标题
                    "file_title": "string", # 文件名称
                    "content": "string",  # 内容(title+body)
                },
                {
                    "title": "string",  # 标题
                    "parent_title": "string", # 父标题
                    "file_title": "string", # 文件名称
                    "content": "string",  # 内容(title+body)
                }]

            5.日志统计

            6.备份chunks

        :param state:
        :return:
        """

        #1. 获取输入参数并校验
        md_content, file_title, max_content_length, min_content_length = self._validate_get_input(state)

        #2. 按照标题切分
        sections:List[dict[str,Any]] = self._split_by_title(md_content, file_title)

        #3. 拆分或合并
        final_sections = self._split_long_merge_short(sections, max_content_length, min_content_length)

        #4. 组装chunk(将body换成content(标题+body))
        final_chunks = self._assemble_chunk(final_sections)

        #5. 日志&备份
        self._log_summary(md_content, final_chunks, max_content_length)
        self._backup_chunks(state, final_chunks)

        state["chunks"] = final_chunks

        return state

    def _validate_get_input(self, state):

        self.log_step("step1", "获取输入参数并校验")
        md_content = state.get("md_content")
        file_title = state.get("file_title")
        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length

        if not md_content:
            raise ValidationError("切分的文档不存在")
        if not file_title:
            raise ValidationError("参数错误:文件标题为空")
        if max_content_length < 0:
            raise ValidationError("参数错误:max_content_length小于0")
        if min_content_length < 0:
            raise ValidationError("参数错误:min_content_length小于0")
        if max_content_length < min_content_length:
            raise ValidationError("参数错误:max_content_length小于min_content_length")

        return md_content, file_title, max_content_length, min_content_length

    def _split_by_title(self, md_content, file_title) -> List[dict[str, Any]]:

        self.log_step("step2", "按照标题切分")
        sections : List[dict[str, Any]] = []
        body_lines = []
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
        title = ""
        hierarchy = [""]*7      #存放标题的层级，用于判断标题的父子关系    ["","# 标题","## 标题","### 标题","#### 标题","##### 标题","###### 标题"]
        current_level = 0
        is_fence = False

        # 封装section
        def _flush():
            body = "\n".join(body_lines)
            if body:        #有些md文档开头的段落没有标题，做分类处理

                parent_title = ""
                for i in range(current_level-1,0,-1):
                    parent_title = hierarchy[i] if hierarchy[i] else ""
                    if parent_title: break
                if not parent_title:
                    parent_title = parent_title if title else file_title

                section = {
                "title": title if title else file_title,
                "parent_title": parent_title,
                "file_title": file_title,
                "body": body
                }
                sections.append(section)
            else:
                pass


        #1. 获取所有行集合
        lines = md_content.split("\n")

        #2. 遍历行集合，判断是否存在代码块，正则表达式识别标题，构建section
        for line in lines:

            # 判断是否存在代码块
            if line.strip().startswith("```") or line.strip().startswith("~~~"):
                is_fence = not is_fence

            match = heading_re.match(line)
            if match and not is_fence:
                _flush()
                title = line
                level = len(match.group(1))     # 获取#数量，判断标题级别
                current_level = level
                hierarchy[level] = line
                for i in range(level+1, 7,1):       #清除当前标题级别之后的标题
                    hierarchy[i] = ""
                body_lines = []     #清除上个section的body_lines
            else:
                body_lines.append(line)
        _flush()

        return sections

    def _split_long_merge_short(self, sections, max_content_length=1000, min_content_length=200) -> List[dict[str, Any]]:

        self.log_step("step3", "拆分或合并")
        current_sections = []
        for section in sections:
            sub_sections = self._split_long(section,max_content_length)
            current_sections.extend(sub_sections)       #append就把列表加进列表里了，extend是把列表元素加进列表里了

        final_sections = self._merge_short(current_sections,min_content_length)
        return final_sections

    def _split_long(self, section:dict[str, Any], max_content_length=1000) -> List[dict[str, Any]]:

        self.log_step("step3.1", "拆分长段落")
        sub_section = []

        # 1.获取section对象属性
        title = section.get("title", "")
        parent_title = section.get("parent_title", "")
        file_title = section.get("file_title", "")
        body = section.get("body", "")

        # 2.判断表格
        #利用工具类将表格降维处理
        if "<table>" in body:
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        # 3.对标题校验长度，超过50截断
        if len(title) > 50:
            title = title[:50]

        # 4.拼接标题前缀
        title_prefix = f" {title}\n\n"

        # 5.计算标题前缀长度 + 内容长度
        content_length = len(title_prefix) + len(body)

        # 6.判断是否需要切分
        if content_length <= max_content_length:
            return [section]

        # 7.计算body可用长度，判断长度是否小于等于0
        body_available_length = max_content_length - len(title_prefix)
        if body_available_length <= 0:
            return [section]

        # 8.需要继续切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_content_length,  # 单个切分内容最大长度
            chunk_overlap=0,  # 切分内容重叠部分长度
            keep_separator=False,  # 是否保留分隔符
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
        )
        texts: List[str] = splitter.split_text(body)
        sub_section = []
        for index,text in enumerate(texts):
            sub_section.append({
                "title":f"{title}",
                "parent_title": parent_title,
                "file_title": file_title,
                "body": text,
                "part": f"{index + 1}"  # 标记序号
            })

        #     判断切分数量结果，要么返回唯一原元素列表，要么返回多个新元素类别

        # 9.返回
        return sub_section

    def _merge_short(self, current_sections, min_content_length=100) -> List[dict[str, Any]]:
        """
        贪心累加算法
            1.累加过程可能会超过阈值，可接受
            2.孤儿小块别忘记处理
        :param current_sections: 当前section列表
        :param min_content_length: 最小内容长度
        :return: 合并后的section列表
        """

        self.log_step("step3.2", "合并短段落")
        final_sections = []
        current_section = current_sections[0]

        for next_section in current_sections[1:]:

            #同一个大章节下，两个小章节可以合并
            same_title = current_section.get("parent_title") == next_section.get("parent_title")
            if same_title and len(current_section.get("body")) < min_content_length:
                current_section["body"] = current_section["body"].rstrip() + "\n\n" + next_section["body"].lstrip()
                current_section["title"] = current_section["parent_title"]
            else:
                final_sections.append(current_section)
                current_section = next_section

        final_sections.append(current_section)

        # 对所有 section 的 part 做处理
        part_counter = {}
        result = []
        for final_section in final_sections:
            if "part" in final_section:
                parent_title = final_section.get('parent_title')
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                final_section['part'] = new_part
                final_section['title'] = final_section['title'] + f"- {new_part}"

            result.append(final_section)

        return result

    def _assemble_chunk(self, final_sections) -> List[Dict[str, Any]]:

        self.log_step("step4", "组装切片")
        final_chunks = []

        for section in final_sections:
            chunk = {
                "title": section.get("title", ""),
                "parent_title": section.get("parent_title", ""),
                "file_title": section.get("file_title", ""),
                "content": section["title"] +"\n\n" + section["body"]
            }

            if "part" in chunk:
                chunk['part'] = section.get('part')

            final_chunks.append(chunk)

        return final_chunks

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)        #注意不是json.dumps，而是json.dump，后者有写的能力
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()
    # 构造状态字典
    file_path = r"D:\temp_dir\万用表RS-12的使用\万用表RS-12的使用_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "万用表RS-12的使用_new",
        "md_content": content,
        "file_dir": r"D:\temp_dir\万用表RS-12的使用"
    }
    document_node.process(state)