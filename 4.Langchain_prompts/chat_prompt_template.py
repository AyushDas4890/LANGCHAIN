from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()


chat_template = ChatPromptTemplate([  ## ChatPromptTemplate is used for multi turn messages to create dynmic prompt, while prompt template is used for single turn dynamic messages.
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms , what is {topic}")

])


prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'LBW'})

result = model.invoke(prompt)
print(result)

