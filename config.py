from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "male_players.csv"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "faiss_player_index"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULT_TOP_K = 5
MAX_INDEX_ROWS = 5000