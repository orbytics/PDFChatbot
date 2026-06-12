import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import config


def build_index() -> FAISS:
    pdf = fitz.open(config.PDF_PATH)
    raw_docs = []
    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text()
        if text.strip():
            raw_docs.append(Document(
                page_content=text,
                metadata={"page": page_num, "source": config.PDF_PATH},
            ))
    pdf.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
    return FAISS.from_documents(chunks, embeddings)
