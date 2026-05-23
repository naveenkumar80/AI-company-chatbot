import path from "node:path";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { fileURLToPath } from "node:url";
import { answerQuestion, getDefaultPdfPath, indexTheDocs } from "./prepare.js";

// __dirname is not available in ES modules, so recreate it from import.meta.url.
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  const [command = "help", ...args] = process.argv.slice(2);

  // Build or refresh the vector store from a PDF.
  if (command === "index") {
    // Default to the project PDF, but allow `node rag.js index ./other.pdf`.
    const pdfPath = args[0] ? path.resolve(args[0]) : getDefaultPdfPath(__dirname);
    await indexTheDocs(pdfPath);
    return;
  }

  // Ask one question and exit. Useful for quick tests and scripts.
  if (command === "ask") {
    // Join all remaining CLI words so quoted and unquoted questions both work.
    const question = args.join(" ").trim();
    if (!question) {
      throw new Error('Usage: node rag.js ask "your question"');
    }

    console.log(await answerQuestion(question));
    return;
  }

  // Start a simple terminal chat loop.
  if (command === "chat") {
    await startChat();
    return;
  }

  printHelp();
}

async function startChat() {
  const rl = readline.createInterface({ input, output });
  console.log("Company chatbot ready. Type a question, or 'exit' to quit.");

  // Keep asking until the user presses Enter on an empty line or types "exit".
  while (true) {
    const question = (await rl.question("> ")).trim();
    if (!question || question.toLowerCase() === "exit") {
      break;
    }

    try {
      console.log(await answerQuestion(question));
    } catch (error) {
      console.error(`Error: ${error.message}`);
    }
  }

  rl.close();
}

function printHelp() {
  console.log(`
Usage:
  node rag.js index [pdfPath]        Index the PDF into Pinecone
  node rag.js ask "question"         Ask one question using retrieved context
  node rag.js chat                   Start an interactive chat

Environment:
  GEMINI_API_KEY or GOOGLE_API_KEY   Required for embeddings and Pinecone retrieval
  GROQ_API_KEY                       Groq API key for chat
  GROQ_CHAT_MODEL                    Optional chat model, defaults to llama-3.1-8b-instant
  PINECONE_API_KEY                   Pinecone API key
  PINECONE_INDEX                     Pinecone index name
  PINECONE_NAMESPACE                 Optional namespace, defaults to company-docs
`);
}

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  process.exitCode = 1;
});
