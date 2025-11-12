from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel


### RunnableParallel  is a runnable primitive that allows multiple runnables to execute in parllel. Each runnable receives the sae input and processes it independently, producing a dictionary of outputs.


# Load environment variables
load_dotenv()

# Define prompts
prompt1 = PromptTemplate(
    template='Write a tweet on the topic: {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Write a linkedin post on the topic: {topic}',
    input_variables=['topic']
)

# Initialize model and parser
model = ChatOpenAI()
parser = StrOutputParser()

# Define parallel chain correctly
parallel_chain = RunnableParallel({
    "tweet": prompt1 | model | parser,
    "linkedin_post": prompt2 | model | parser
})

# Invoke the chain
result = parallel_chain.invoke({'topic': 'Artificial Intelligence'})

# Print result
print(result)
