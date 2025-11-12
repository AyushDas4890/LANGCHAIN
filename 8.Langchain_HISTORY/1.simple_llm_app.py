from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables (make sure OPENAI_API_KEY is in your .env file)
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Create a prompt template
prompt = PromptTemplate(
    template=(
        "You are a creative AI assistant.\n"
        "1. Suggest a catchy blog title about {topic}."
        
    ),
    input_variables=["topic"]
)

# Get input from user
topic = input("Enter a topic: ")

# Format the prompt
formatted_prompt = prompt.format(topic=topic)

# Generate response from the LLM
response = llm.invoke(formatted_prompt)

# Print output
print("\n=== AI Generated Content ===\n")
print(response.content)
