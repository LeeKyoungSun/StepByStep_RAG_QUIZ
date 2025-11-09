from pathlib import Path
from dotenv import load_dotenv

# .env 파일 절대경로 지정 (가장 먼저 로드)
env_file = Path(__file__).parent.parent / ".env"
load_dotenv(env_file)
