from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

### RunnableSequence is a sequential chain of runnables in Langchain that executes each step one after another, passing the output of one step as the input to the next
load_dotenv()  # Load environment variables from .env file

prompt1 = PromptTemplate(
    template='Write a joke ablout {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Explain the following joke: {text}',
    input_variables=['text']
)

model = ChatOpenAI()
parser = StrOutputParser()

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result = chain.invoke({'topic':'AI'})

print(result)

