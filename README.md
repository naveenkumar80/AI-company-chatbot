# AI Company Chatbot

A Retrieval Augmented Generation (RAG) chatbot for a company PDF. It indexes a PDF into Pinecone, retrieves relevant context for a user question, and answers with Groq.

## Architecture

1. `rag.js` is the command-line entry point.
2. `prepare.js` contains the RAG logic.
3. Gemini embeddings convert PDF chunks and user questions into vectors.
4. Pinecone stores and searches the vectors.
5. Groq generates the final chat answer from the retrieved context.
6. `.vector-store/company-docs.json` stores a local text manifest for debugging and fallback retrieval.

## Flow

Indexing:

1. Load `AI_Company_Chatbot_Detailed_Docs.pdf`.
2. Split the PDF text into overlapping chunks.
3. Generate Gemini embeddings for each chunk.
4. Upload the chunks to Pinecone.
5. Save a local manifest without embedding vectors.

Chat:

1. Embed the user question with Gemini.
2. Search Pinecone for the most relevant chunks.
3. Send the retrieved context and question to Groq.
4. If Gemini embeddings fail, use local keyword retrieval from the manifest so chat can still run.

## Setup

Install dependencies:

```bash
npm install
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=your_pinecone_index_name

# Optional
PINECONE_NAMESPACE=company-docs
GEMINI_EMBEDDING_MODEL=text-embedding-004
GROQ_CHAT_MODEL=llama-3.1-8b-instant
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=4
```

Your Pinecone index must use the same vector dimension as `GEMINI_EMBEDDING_MODEL`, and the index must belong to the same Pinecone project as `PINECONE_API_KEY`.

## Commands

Index the default PDF:

```bash
npm run index
```

Index another PDF:

```bash
npm run index -- ./path/to/document.pdf
```

Ask one question:

```bash
npm run ask -- "what is the mission of company"
```

Start interactive chat:

```bash
npm run chat
```

Run syntax checks:

```bash
npm run check
```

## Files

- `prepare.js`: PDF loading, chunking, Gemini embeddings, Pinecone storage, Groq answering, and fallback retrieval.
- `rag.js`: CLI commands for `index`, `ask`, and `chat`.
- `index.js`: compatibility wrapper that forwards to `rag.js`.
- `.vector-store/company-docs.json`: local manifest used for debugging and keyword fallback.
- `AI_Company_Chatbot_Detailed_Docs.pdf`: default company document.

## PowerShell Notes

To read environment variables in PowerShell:

```powershell
$env:PINECONE_INDEX
```

`process.env.PINECONE_INDEX` is JavaScript syntax and only works inside Node.js code.

## Troubleshooting

`Gemini embeddings failed. Falling back to local keyword retrieval.`

The app is still usable, but semantic Pinecone retrieval is unavailable. Fix `GEMINI_API_KEY` or `GOOGLE_API_KEY`, then run `npm run index` again.

`Pinecone rejected PINECONE_API_KEY`

Check that `PINECONE_API_KEY` and `PINECONE_INDEX` belong to the same Pinecone project.

`Set GROQ_API_KEY`

Add a valid Groq API key to `.env`; Groq is used for the final chat response.
