import os
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from gemini_client import text,embedding
load_dotenv()

pc=Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)
""" index_name="company-policies"
pc.create_index(
    name=index_name,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
) """
print(pc.list_indexes())

index=pc.Index("company-policies")

index.upsert(
    vectors=[
        {
            "id":"chunk_001",
            "values": embedding,
            "metadata":{
                "text":text
            }
        }
    ]
)
print("vector upserted successfully")
