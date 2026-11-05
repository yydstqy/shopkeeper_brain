from dotenv import load_dotenv; load_dotenv(r"D:\PyCharm 2026.2.3\pyprojcet\shopkeeper_brain\knowledge\.env")
import os
print(repr(os.getenv("BGE_DEVICE")), repr(os.getenv("BGE_FP16")))