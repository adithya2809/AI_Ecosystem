# Company Policy RAG System

This project demonstrates a complete retrieval-augmented generation (RAG) flow for answering questions about company policies.

The system combines:

- Google Gemini for text embeddings and answer generation
- Pinecone for vector storage and similarity search
- Python for orchestration
- Environment variables for API credentials

## What RAG Means Here

The application does not ask Gemini to answer from its general knowledge alone. It first retrieves the most relevant policy text from Pinecone and then gives that text to Gemini as context:

```text
Policy text
    -> Gemini embedding
    -> Pinecone vector index
    -> semantic search for a user query
    -> retrieved policy context
    -> Gemini-generated answer
```

For the current demonstration, the indexed policy text is:

> Employees recieve 20 days of annual leave

The sample question is:

> How many vacation days do the employees get?

## Build Flow From Git History

The commit history shows the system being built incrementally:

| Commit | Stage | What was added |
| --- | --- | --- |
| `708f850` | Embedding foundation | Added `gemini_client.py`, configured the Google GenAI client, embedded policy text with `gemini-embedding-2`, and pinned `google-genai` in `requirements.txt`. |
| `d49647e` | Vector database | Added creation of the `company-policies` Pinecone index with 768 dimensions and cosine similarity. |
| `d35cb3b` | Indexing | Connected the generated embedding and source text, then upserted them into Pinecone as `chunk_001`. |
| `2ffd2ae` | Retrieval | Embedded the user query with the same 768-dimensional model and retrieved the nearest stored vector with `top_k=1`. |
| `4d648dc` | Generation | Added a prompt containing the retrieved text and generated the final answer with `gemini-3.5-flash-lite`. |

The earlier commits `22113d6` and `0dbb7e9` also established the credential boundary by ignoring `.env` and removing it from version control.

## Architecture

### 1. Embedding the source text

[`gemini_client.py`](gemini_client.py) creates a Google GenAI client and converts the policy text into a numerical vector:

```python
result = client.models.embed_content(
    model="gemini-embedding-2",
    contents=text,
    config=types.EmbedContentConfig(
        output_dimensionality=768
    )
)
embedding = result.embeddings[0].values
```

The vector has 768 dimensions, matching the Pinecone index configuration.

### 2. Storing the policy in Pinecone

[`main.py`](main.py) connects to Pinecone and opens the `company-policies` index. The indexing block stores both the vector and the original text as metadata:

```python
index.upsert(
    vectors=[
        {
            "id": "chunk_001",
            "values": embedding,
            "metadata": {"text": text}
        }
    ]
)
```

Keeping the original text in metadata lets the application use the retrieved passage when constructing the generation prompt.

### 3. Retrieving relevant context

When a question arrives, the application embeds the question using the same Gemini embedding model and searches Pinecone:

```python
result = index.query(
    vector=query_embedding,
    top_k=1,
    include_metadata=True
)
```

The nearest match is selected and its stored policy text is extracted:

```python
retrieved_text = result["matches"][0]["metadata"]["text"]
```

### 4. Generating a grounded answer

The retrieved text and the user question are combined into a prompt that instructs Gemini to answer only from the supplied context:

```python
prompt = f"""
    Answer the user's question using only the provided context
    context: {retrieved_text}
    Question: {query}
"""
```

Gemini then produces the final response with `gemini-3.5-flash-lite`.

## Setup

Create a `.env` file in the project directory with the credentials used by the clients:

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Install the pinned dependencies into the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The Pinecone index must be named `company-policies`, use cosine similarity, and have a dimension of `768`.

## Run the Demo

```powershell
.\.venv\Scripts\python.exe main.py
```

The script connects to Pinecone, embeds the query, retrieves the closest policy chunk, and prints the generated answer.

## Current Scope

This is a complete working RAG prototype for a single policy chunk. The current implementation demonstrates every core RAG stage, but it is intentionally small:

- The source text is hard-coded rather than loaded from documents.
- The indexing and index-creation blocks are commented after their one-time use.
- Retrieval currently requests one result with `top_k=1`.
- There is no chunking, document loader, reranking, web API, conversation memory, or automated evaluation yet.
- The prompt asks for grounded answers, but the application does not yet implement an explicit no-context fallback or citation display.

Those are natural next steps for turning this proof of concept into a multi-document production system.

## Result

The final pipeline is:

1. Embed company policy text with Gemini.
2. Store the embedding and source text in Pinecone.
3. Embed the user's question.
4. Retrieve the most similar policy chunk.
5. Pass the retrieved context to Gemini.
6. Return a context-grounded answer.
