import fs from "node:fs/promises";
import path from "node:path";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import Groq from "groq-sdk";
import dotenv from "dotenv";

// Load secrets and model settings from .env before any clients are created.
dotenv.config();

// Local manifest path. Pinecone is the real vector DB, but this JSON file helps with debugging
// and gives the chatbot a backup retrieval path when Gemini embeddings are unavailable.
const VECTOR_STORE_DIR = ".vector-store";
const VECTOR_STORE_PATH = path.join(VECTOR_STORE_DIR, "company-docs.json");
const DEFAULT_PDF_PATH = "AI_Company_Chatbot_Detailed_Docs.pdf";
const DEFAULT_NAMESPACE = "company-docs";

// Tuning values can be changed in .env without editing code.
const CHUNK_SIZE = Number(process.env.CHUNK_SIZE || 500);
const CHUNK_OVERLAP = Number(process.env.CHUNK_OVERLAP || 100);
const TOP_K = Number(process.env.TOP_K || 4);
const EMBEDDING_MODEL = process.env.GEMINI_EMBEDDING_MODEL || "text-embedding-004";
const CHAT_MODEL = process.env.GROQ_CHAT_MODEL || "llama-3.1-8b-instant";

// These words are ignored only by the local keyword fallback. Semantic Pinecone retrieval
// still uses the full user question.
const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "about",
  "for",
  "is",
  "of",
  "our",
  "the",
  "to",
  "what",
]);

export function getDefaultPdfPath(projectDir = process.cwd()) {
  return path.join(projectDir, DEFAULT_PDF_PATH);
}

// Stage 1 of RAG: load the PDF, split it into chunks, embed each chunk, and upload to Pinecone.
export async function indexTheDocs(pdfPath = getDefaultPdfPath()) {
  const resolvedPdfPath = path.resolve(pdfPath);
  console.log(`Indexing documents from ${resolvedPdfPath}`);

  // Load the whole PDF first, then let the splitter create retrieval-sized chunks.
  const loader = new PDFLoader(resolvedPdfPath, {
    splitPages: false,
  });
  const docs = await loader.load();

  // Overlap helps preserve context when an answer spans two nearby chunks.
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
  });
  const chunks = await splitter.splitDocuments(docs);

  if (chunks.length === 0) {
    throw new Error("No text chunks were created from the PDF.");
  }

  console.log(`Chunks created: ${chunks.length}`);

  const vectorStore = await getVectorStore();
  const sourceName = path.basename(resolvedPdfPath);

  // Pinecone metadata should stay flat and small; nested PDF metadata can fail on upsert.
  const documents = chunks.map((chunk, index) => ({
    pageContent: chunk.pageContent,
    metadata: {
      source: sourceName,
      chunkIndex: index,
    },
  }));
  const ids = documents.map((document) => `${document.metadata.source}-${document.metadata.chunkIndex}`);

  // Reusing deterministic ids makes repeated indexing update the same records instead of duplicating them.
  await runWithHelpfulProviderErrors(() => vectorStore.addDocuments(documents, { ids }));

  await writeLocalManifest({
    pdfPath: resolvedPdfPath,
    documents,
    ids,
  });

  console.log(`Uploaded ${documents.length} chunks to Pinecone.`);
  console.log(`Local manifest saved to ${VECTOR_STORE_PATH}`);

  return documents;
}

// Stage 2 of RAG: retrieve relevant context, then ask Groq to answer using only that context.
export async function answerQuestion(question) {
  if (!question.trim()) {
    throw new Error("Ask a non-empty question.");
  }

  // Prefer Pinecone semantic retrieval. If Gemini embeddings are misconfigured,
  // fall back to keyword search over the local manifest so Groq chat can still run.
  const context = await getRetrievedContext(question);

  // Groq handles the chat generation step; Gemini is still used only for embeddings.
  const response = await getGroqClient().chat.completions.create({
    model: CHAT_MODEL,
    temperature: 0.2,
    messages: [
      {
        role: "system",
        content:
          'You are an AI company chatbot. Answer using only the provided company context. If the context does not contain the answer, say: "I don\'t know based on the provided company documents." Include source names when they help.',
      },
      {
        role: "user",
        content: `Context:\n${context}\n\nQuestion:\n${question}`,
      },
    ],
  });

  return response.choices[0]?.message?.content || "I could not generate an answer.";
}

// Optional helper for code that wants a LangChain retriever instead of this CLI's answer flow.
export async function createRetriever() {
  const vectorStore = await getVectorStore();
  return vectorStore.asRetriever({ k: TOP_K });
}

async function getRetrievedContext(question) {
  try {
    const vectorStore = await getVectorStore();

    // Semantic search embeds the question, then asks Pinecone for nearby document chunks.
    const matches = await runWithHelpfulProviderErrors(() =>
      vectorStore.similaritySearchWithScore(question, TOP_K),
    );
    return formatRetrievedContext(matches);
  } catch (error) {
    if (!isGeminiApiKeyError(error)) {
      throw error;
    }

    console.warn("Warning: Gemini embeddings failed. Falling back to local keyword retrieval.");
    return formatLocalKeywordContext(await retrieveFromLocalManifest(question));
  }
}

async function getVectorStore() {
  // The embedding model used here must match the dimension of your Pinecone index.
  const embeddings = new GoogleGenerativeAIEmbeddings({
    model: EMBEDDING_MODEL,
    apiKey: getGoogleApiKey(),
  });

  const pineconeIndex = getPineconeIndex();

  // LangChain expects an actual Pinecone Index object here, not the index name string.
  return PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex,
    namespace: process.env.PINECONE_NAMESPACE || DEFAULT_NAMESPACE,
    maxConcurrency: 5,
  });
}

function getPineconeIndex() {
  const apiKey = process.env.PINECONE_API_KEY;
  const indexName = process.env.PINECONE_INDEX;

  if (!apiKey) {
    throw new Error("Set PINECONE_API_KEY in your .env file.");
  }

  if (!indexName) {
    throw new Error("Set PINECONE_INDEX in your .env file. In PowerShell, read it with `$env:PINECONE_INDEX`.");
  }

  const pinecone = new PineconeClient({ apiKey });
  return pinecone.Index(indexName);
}

function getGoogleApiKey() {
  // LangChain documents GOOGLE_API_KEY; many Gemini tutorials use GEMINI_API_KEY.
  const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;

  if (!apiKey) {
    throw new Error("Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.");
  }

  return apiKey;
}

function getGroqClient() {
  const apiKey = process.env.GROQ_API_KEY;

  if (!apiKey) {
    throw new Error("Set GROQ_API_KEY in your .env file before asking questions.");
  }

  return new Groq({ apiKey });
}

async function writeLocalManifest({ pdfPath, documents, ids }) {
  // This manifest intentionally stores text and metadata, not full embedding vectors.
  // It is small enough to inspect and can power a basic local fallback search.
  await fs.mkdir(VECTOR_STORE_DIR, { recursive: true });
  await fs.writeFile(
    VECTOR_STORE_PATH,
    JSON.stringify(
      {
        createdAt: new Date().toISOString(),
        source: pdfPath,
        embeddingModel: EMBEDDING_MODEL,
        chatModel: CHAT_MODEL,
        pineconeIndex: process.env.PINECONE_INDEX,
        pineconeNamespace: process.env.PINECONE_NAMESPACE || DEFAULT_NAMESPACE,
        records: documents.map((document, index) => ({
          id: ids[index],
          text: document.pageContent,
          metadata: document.metadata,
        })),
      },
      null,
      2,
    ),
  );
}

function formatRetrievedContext(matches) {
  if (!matches.length) {
    return "No relevant context was retrieved.";
  }

  return matches
    .map(([document, score], index) => {
      const source = document.metadata?.source || "unknown source";
      const chunkIndex = document.metadata?.chunkIndex ?? "unknown";
      return `[${index + 1}] Source: ${source}, chunk: ${chunkIndex}, score: ${score.toFixed(4)}
${document.pageContent}`;
    })
    .join("\n\n---\n\n");
}

async function retrieveFromLocalManifest(question) {
  const raw = await fs.readFile(VECTOR_STORE_PATH, "utf8").catch(() => null);
  if (!raw) {
    throw new Error(
      "Local manifest not found. Fix GEMINI_API_KEY or GOOGLE_API_KEY, then run `node rag.js index` before chatting.",
    );
  }

  const manifest = JSON.parse(raw);
  const records = Array.isArray(manifest.records) ? manifest.records : [];
  if (records.length === 0) {
    throw new Error("Local manifest has no records. Run `node rag.js index` again.");
  }

  // Filter weak words so questions like "what is the mission of company" focus on "mission".
  const queryTerms = tokenize(question).filter((term) => !STOP_WORDS.has(term));
  const scoredRecords = records
    .map((record) => ({
      record,
      score: keywordScore(queryTerms, record.text || ""),
    }))
    .sort((a, b) => b.score - a.score)
    .filter(({ score }) => score > 0)
    .slice(0, TOP_K);

  return scoredRecords;
}

function formatLocalKeywordContext(matches) {
  if (!matches.length) {
    return "No relevant context was retrieved.";
  }

  return matches
    .map(({ record, score }, index) => {
      const source = path.basename(record.metadata?.source || record.id || "local manifest");
      return `[${index + 1}] Source: ${source}, score: ${score.toFixed(4)}
${record.text}`;
    })
    .join("\n\n---\n\n");
}

function keywordScore(queryTerms, text) {
  const textTerms = tokenize(text);
  if (queryTerms.length === 0 || textTerms.length === 0) {
    return 0;
  }

  const frequencies = new Map();
  for (const term of textTerms) {
    frequencies.set(term, (frequencies.get(term) || 0) + 1);
  }

  // Add small boosts for exact phrases and section headings. This makes fallback retrieval
  // good enough for demos even though Pinecone semantic search is still the preferred path.
  const normalizedText = String(text).toLowerCase();
  const exactPhraseBoost = normalizedText.includes(queryTerms.join(" ")) ? 3 : 0;
  const sectionTitleBoost = queryTerms.some((term) =>
    normalizedText.includes(`\n${term}`) || normalizedText.startsWith(term),
  )
    ? 2
    : 0;
  const termHits = queryTerms.reduce((score, term) => score + (frequencies.get(term) || 0), 0);

  return termHits + exactPhraseBoost + sectionTitleBoost;
}

function tokenize(text) {
  return String(text)
    .toLowerCase()
    .match(/[a-z0-9]+/g) || [];
}

function isGeminiApiKeyError(error) {
  return (
    String(error.message).includes("Gemini rejected your embedding API key") ||
    String(error.message).includes("API key not valid")
  );
}

async function runWithHelpfulProviderErrors(action) {
  try {
    return await action();
  } catch (error) {
    if (error.name === "PineconeAuthorizationError") {
      throw new Error(
        `Pinecone rejected PINECONE_API_KEY while opening index "${process.env.PINECONE_INDEX}". Check that the API key and index belong to the same Pinecone project.`,
      );
    }

    if (String(error.message).includes("API key not valid")) {
      throw new Error(
        "Gemini rejected your embedding API key. Chat uses Groq, but retrieval still uses Gemini embeddings, so update GEMINI_API_KEY or GOOGLE_API_KEY in `.env` and restart `node rag.js chat`.",
      );
    }

    throw error;
  }
}
