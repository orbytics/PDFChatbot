from langchain_google_genai import ChatGoogleGenerativeAI
import config


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        temperature=config.TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY,
    )
