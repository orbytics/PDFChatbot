from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from src.retriever import get_retriever
from src.llm import get_llm

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and the latest user question, "
     "reformulate the question so it is fully self-contained and understandable "
     "without the conversation history. "
     "Do NOT answer the question. "
     "Only rewrite it if it references prior context; otherwise return it as-is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant that answers questions strictly from the "
     "context provided below. Do not use any outside knowledge.\n\n"
     "If the answer is not explicitly present in the context, respond with exactly:\n"
     "\"I do not know the answer based on the provided documentation.\"\n\n"
     "Do not extrapolate, infer, summarise beyond the text, or guess.\n"
     "Cite page numbers inline as [Page N].\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def build_chain(index: FAISS):
    llm = get_llm()
    retriever = get_retriever(index)
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_PROMPT
    )
    qa_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    return create_retrieval_chain(history_aware_retriever, qa_chain)
