### RunnablePassthrough is a simple runnable in Langchain that returns the input it receives without any modifications or processing.


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel
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

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explaination": RunnableSequence(prompt2,model,parser)
})


chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

result = chain.invoke({'topic':'AI'})

print(result)

