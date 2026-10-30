from functools import cache, lru_cache

from knowledge.service.import_file_service import ImportFileService


#专业创建对象，并进行缓存管理，可以实现对象共享
@cache      #缓存满了不清理    OOM问题(内存溢出)
# @lru_cache      #缓存满了可以根据lru(最近最少使用)策略进行缓存清理
def get_import_file_service() -> ImportFileService:
    return ImportFileService()