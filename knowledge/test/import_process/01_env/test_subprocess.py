"""
  @Author:tqy
  @Time:2026/10/16
  @Desc:
"""
import os
import subprocess
from pathlib import Path

from sympy import true

# 基础用法：run() - 同步执行，等待完成
os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'#设置MinerU的模型来源 ModelScope（阿里魔搭）
os.environ['MODELSCOPE_OFFLINE'] = '1'#1 表示开启离线模式（MODELSCOPE_CACHE 指定的目录）
os.environ['HF_HUB_OFFLINE'] = '1'#启用 HuggingFace Hub 离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'#启用 Transformers 库离线模式

print("✅ 环境变量已配置")
print(f"   MINERU_MODEL_SOURCE = {os.environ.get('MINERU_MODEL_SOURCE')}")
print(f"   MODELSCOPE_OFFLINE = {os.environ.get('MODELSCOPE_OFFLINE')}")
print(f"   HF_HOME = {os.environ.get('HF_HOME')}")
print()

#   mineru-kit parse D:\doc\万用表RS-12的使用.pdf -o D:\temp_dir --tier standard
result = subprocess.run(
    [
        "mineru-kit",
        "parse",
        r"D:\doc\万用表RS-12的使用.pdf",
        "-o",
        r"D:\temp_dir",
        "--tier",
        "standard"
    ],
    capture_output=True,  # 捕获输出
    encoding='utf-8',  # 指定 UTF-8 编码（Windows 下避免 GBK 问题）
    errors='replace',  # 遇到无法解码的字符用  替换
    check=False,  # 不自动抛异常，手动检查
    shell=True
)

print(result)





# 高级用法：Popen() - 实时获取输出
# ===============================================================

# === 关键：加载环境变量（模拟 PyCharm 运行 pdf_to_md.py 时的环境）===
# 方法1：直接读取 .env 文件并设置环境变量
# env_file = Path(__file__).parent.parent.parent / '.env'
# if env_file.exists():
#     with open(env_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line and not line.startswith('#') and '=' in line:
#                 key, value = line.split('=', 1)
#                 os.environ[key.strip()] = value.strip()

# 方法2：显式设置关键环境变量（确保 MinerU 使用 ModelScope 离线模式）
os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'
os.environ['MODELSCOPE_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("✅ 环境变量已配置")
print(f"   MINERU_MODEL_SOURCE = {os.environ.get('MINERU_MODEL_SOURCE')}")
print(f"   MODELSCOPE_OFFLINE = {os.environ.get('MODELSCOPE_OFFLINE')}")
print(f"   HF_HOME = {os.environ.get('HF_HOME')}")
print()

proc = subprocess.Popen(
    args=["mineru-kit", "parse", r"D:\doc\万用表RS-12的使用.pdf",
          "-o", r"D:\temp_dir", "--tier", "standard","--format","zip"],
    stdout=subprocess.PIPE,    # 捕获标准输出
    stderr=subprocess.STDOUT,  # 合并错误到标准输出
    text=True,
    encoding="utf-8",
    errors="replace",          # 遇到乱码时替换
    bufsize=1                  # 行缓冲，实时输出
)

for line in proc.stdout:
    print(line.rstrip())

return_code = proc.wait()
print(return_code)