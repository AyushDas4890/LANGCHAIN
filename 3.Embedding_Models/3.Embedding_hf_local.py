from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="google/gemma-2-2b-it")  

text = "Delhi is the capital of India"
result = embedding.embed_query(text)

print(str(result))