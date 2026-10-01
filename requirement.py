from gemini_client import client
from google.genai import types
from pinecone import Pinecone,ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()
activities = [
    {
        "id": "1",
        "name": "Paragliding",
        "description": "Glide over the sea with thrill and fun",
        "keywords": ["adventure", "beach", "water"],
        "destination": "Goa",
        "cost": 2000
    },
    {
        "id": "2",
        "name": "Cultural Heritage Tour",
        "description": "Visit historic Portuguese architecture",
        "keywords": ["culture", "history", "explore"],
        "destination": "Goa",
        "cost": 1000
    },
    {
        "id": "3",
        "name": "Beachside Sunset Kayaking",
        "description": "Enjoy a peaceful sunset kayaking experience along the coast",
        "keywords": ["nature", "relaxation", "water", "adventure"],
        "destination": "Goa",
        "cost": 1500
    }
]
pc= Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
index_name="activities"
""" pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    ) """
index=pc.Index("activities")

for activity in activities:
    activity_text=f"""
    {activity["name"]}.
    {activity["description"]}.
    keywords:{",".join(activity['keywords'])}
    """
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=activity_text,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    embedding = result.embeddings[0].values
    print(activity["name"], "→", len(embedding))

    index.upsert(
        vectors=[
            {
                "id":activity["id"],
                "values":embedding,
                "metadata":{
                    "text":activity_text,
                    "name":activity["name"],
                    "destination":activity["destination"],
                    "cost":activity["cost"]
                }

            }
        ]
    )