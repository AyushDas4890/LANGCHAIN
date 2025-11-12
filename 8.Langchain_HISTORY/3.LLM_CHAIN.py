from langchain.llms import OpenAI 
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

#load the LLM (GPT-3.5)
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

#Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"], # Defines what input is needed
    template="Generate 5 interesting facts about {topic}." # The actual prompt
)

#Create the LLM Chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain with a specific topic:
topic = "Football"
response = chain.run(topic)

print("Response:", response)