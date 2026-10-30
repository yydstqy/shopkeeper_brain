import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("OPENAI_API_KEY", "")
print(f"OPENAI_API_KEY: {key[:8]}...{key[-4:]}" if key else "OPENAI_API_KEY 为空!")