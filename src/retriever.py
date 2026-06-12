from langchain_community.vectorstores import FAISS
import config


def get_retriever(index: FAISS):
    return index.as_retriever(search_kwargs={"k": config.TOP_K})
