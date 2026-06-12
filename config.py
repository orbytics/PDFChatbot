import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH        = "data/document.pdf"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K           = 5
GEMINI_MODEL    = "gemini-2.0-flash"
EMBED_MODEL     = "models/gemini-embedding-001"
TEMPERATURE     = 0.0
GOOGLE_API_KEY  = os.environ["GOOGLE_API_KEY"]
