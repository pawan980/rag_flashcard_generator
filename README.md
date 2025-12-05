***RAG-Powered Flashcard Generator***

AI Flashcards from PDFs, Notes, and Documents — using LangChain + LCEL + Google Gemini: Turn any study material — PDFs, textbooks, lecture notes, scraped webpages — into high-quality, structured flashcards using a Retrieval-Augmented Generation (RAG) pipeline powered by Google Gemini, LangChain, and LCEL.

This project automatically:
|
|--> Loads + chunks documents
|-->Builds a vector database (Chroma + ollama embeddings)
|-->Retrieves the most relevant context
|-->Generates structured flashcards

----------------------------------
🔧 Installation
----------------------------------

1. Clone the repo
git clone https://github.com/pawan980/rag_flashcard_generator.git
cd rag_flashcard-generator

2. Install dependencies
pip install -r requirements.txt


Required packages include:

langchain
langchain-core
langchain-community
langchain-google-genai
langchain-ollama
google-generativeai
chromadb
pypdf
python-dotenv

🔐 Environment Variables
Create a .env file:

GOOGLE_API_KEY=your_gemini_key_here

Get your key from:
➡ https://ai.google.dev

🧠 How It Works
1️⃣ Load & Chunk Documents
vector_db.py handles:
PDF/Text loading
Recursive character chunking
Embedding via Gemini
Storage in ChromaDB
Retriever creation

2️⃣ LCEL Chain for Flashcard Generation

flashcards.py creates a pipeline:

retriever
  | join_docs
  | prompt_template
  | gemini_llm
  | json_parser

3️⃣ Output Formatting

Flashcards are converted into beautiful Markdown:

## 📘 Flashcard 1 — *Topic: LCEL* (Difficulty: Easy)

**Q:** What is LCEL?  
**A:** A pipe-based expression language for composing LangChain workflows.

---

▶️ Usage

To generate flashcards:

python flashcards.py


To generate from custom file:

from flashcards import generate_flashcards

cards = generate_flashcards("my_notes.pdf", pretty=True)
print(cards)

📝 Example Output
## 📘 Flashcard 1 — *Topic: RAG* (Difficulty: Medium)

**Q:** Why is retrieval used before calling an LLM?  
**A:** It reduces hallucination by grounding the model’s response in relevant context.

---

## 📘 Flashcard 2 — *Topic: Embeddings* (Difficulty: Easy)

**Q:** What are embeddings used for?  
**A:** Converting text into vectors for semantic search.
---


🧑‍💻 Author

Pawan
AI Engineer · Python Developer · RAG Architect
✨ Always learning, always building.