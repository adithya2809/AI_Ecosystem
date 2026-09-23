import os
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from gemini_client import text,embedding,client,types
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

""" index.upsert(
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
print("vector upserted successfully") """

query="How many vacation days do the employees get?"
query_result=client.models.embed_content(
    model="gemini-embedding-2",
    contents=query,
    config=types.EmbedContentConfig(
        output_dimensionality=768
    )
)
query_embedding=query_result.embeddings[0].values
result=index.query(
    vector=query_embedding,
    top_k=1,
    include_metadata=True
)
print(result)