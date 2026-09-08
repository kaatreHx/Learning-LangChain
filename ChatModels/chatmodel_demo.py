# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatOpenAI(model='gpt-6-astra')

# res = model.invoke("hi, how are you?")

# print(res)

# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# res = model.invoke("hi, how are you?")

# print(res.content)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

res = model.invoke("what is the capital of india")
print(res.content)