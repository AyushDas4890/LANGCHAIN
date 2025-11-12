from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

document = [
    "Virat Kohli is the captain of Indian Cricket Team.",
    "Rohit Sharma is the vice-captain of Indian Cricket Team.",
    "Sachin Tendulkar is the God of Cricket.",
    "M S Dhoni is the former captain of Indian Cricket Team.",
    "Jasprit Bumrah is the best bowler in the world."
]

query = "tell me about Dhoni"
 

doc_embeddings = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score =sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]


print(query)
print(document[index])

print("Similarity Score: ",score)