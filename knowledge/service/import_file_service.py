"""
  @Author:tqy
  @Time:2026/10/24
  @Desc:
"""
import datetime
import os.path
import shutil
import time
import uuid
import logging

from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.exceptions import FileProcessingError, MinioError
from knowledge.processor.import_process.main_graph import kb_import_graph
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import update_task_status, add_running_task, add_done_task, TASK_STATUS_PROCESSING, \
    TASK_STATUS_FAILED, TASK_STATUS_COMPLETED, add_node_duration

logger = logging.getLogger(__name__)


class ImportFileService:

    """
    业务层:
        1.处理上传文件    双写（上传文件保存到本地，上传文件保存到 minio ）

        2.开启langgraph流程，完成文件导入milvus（7个步骤）
    """

    def run_langgraph_import(self,task_id, file_dir, import_file_path):
        """开启langgraph流程，完成文件导入milvus"""

        update_task_status(task_id, TASK_STATUS_PROCESSING)

        try:
            state = {
                "task_id": task_id,
                "import_file_path": import_file_path,
                "file_dir": file_dir
            }

            final_state = None
            for event in kb_import_graph.stream(state):
                for node_name, state in event.items():
                    print(f"{task_id} | 执行节点名称: {node_name}")
            update_task_status(task_id,TASK_STATUS_COMPLETED)
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)

    def process_upload_file(self,file):
        """
            处理文件上传   双写
            1.创建任务id
            2.设置任务状态："upload_file"任务开始了
            3.上传文件保存到本地
            4.上传文件保存到minio
            5.设置任务状态："upload_file"任务完成了
            6.返回task_id, file_dir, import_file_path
        """

        # 1.创建任务id
        task_id = self._get_task_id()

        #   D:\PyCharm 2026.2.3\pyprojcet\shopkeeper_brain\knowledge\temp_data\20261025\ABCDEFGH
        file_dir = os.path.join(self._get_path_with_date(), task_id)
        start = time.time()

        # 2.设置任务状态："upload_file"任务开始了
        add_running_task(task_id, "upload_file")

        # 3.上传文件保存到本地
        import_file_path = self._save_upload_file_to_local(file, file_dir)

        # 4.上传文件保存到minio
        self._save_upload_file_to_minio(import_file_path,file.filename)

        # 5.设置任务状态："upload_file"任务完成了
        add_done_task(task_id, "upload_file")
        duration = time.time() - start
        add_node_duration(task_id, "upload_file", duration)

        # 6.返回task_id, file_dir, import_file_path
        return task_id, file_dir, import_file_path

    def _get_task_id(self):
        return uuid.uuid4().hex[:8]

    def _get_path_with_date(self):
        # %Y    四位年份    2026
        # %y    两位年份    26
        #D:\PyCharm 2026.2.3\pyprojcet\shopkeeper_brain\knowledge\temp_data\20261024
        return os.path.join(get_local_base_dir(), datetime.datetime.now().strftime("%Y%m%d"))

    def _save_upload_file_to_local(self, file:UploadFile, file_dir):
        """上传文件保存到本地"""
        #   D:\PyCharm 2026.2.3\pyprojcet\shopkeeper_brain\knowledge\temp_data\20261025\ABCDEFGH\万用表RS-12的使用.pdf
        os.makedirs(file_dir, exist_ok=True)
        import_file_path = os.path.join(file_dir, file.filename)
        try:
            with open(import_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)     #   1MB缓存写操作，效率高
        except Exception as e:
            logger.error(f"上传文件保存到本地失败: {e}")
            raise FileProcessingError(f"上传文件保存到本地失败: {e}")

        return import_file_path

    def _save_upload_file_to_minio(self, import_file_path, filename):
        """上传文件保存到minio"""
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"获取minio客户端失败: {e}")
            raise MinioError(f"获取minio客户端失败: {e}")

        MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME","")
        object_name = f"origin/{datetime.datetime.now().strftime('%Y%m%d')}/{filename}"  #   origin/20261025/万用表RS-12的使用.pdf
        try:
            minio_client.fput_object(
                bucket_name=MINIO_BUCKET_NAME   ,
                object_name=object_name,
                file_path=import_file_path
            )
        except Exception as e:
            logger.error(f"上传文件保存到minio失败: {e}")
            raise MinioError(f"上传文件保存到minio失败: {e}")
