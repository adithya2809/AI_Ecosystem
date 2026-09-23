from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client=genai.Client()
text="Employees recieve 20 days of annual leave"

result=client.models.embed_content(
    model="gemini-embedding-2",
    contents=text,
    config=types.EmbedContentConfig(
        output_dimensionality=768
    )
)
embedding=result.embeddings[0].values
print("Dimension:",len(embedding))
print(embedding[:5])