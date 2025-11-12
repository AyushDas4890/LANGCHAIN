from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4')

result = model.invoke("write a short poem of 5 lines")

print(result.content)