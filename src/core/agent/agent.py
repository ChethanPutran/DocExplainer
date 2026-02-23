# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(".env")
from langchain_google_genai import ChatGoogleGenerativeAI
# model = ChatOpenAI(name="gpt-4", temperature=0)

model = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=1.0,  # Gemini 3.0+ defaults to 1.0
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
