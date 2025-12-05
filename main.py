# flashcards.py

import os
from vectordb import create_vector_db
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in your .env file!")

# -----------------------------------------------------
# Gemini LLM
# -----------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY     # <-- key injected here
)

# -----------------------------------------------------
# Flashcard Prompt Template
# -----------------------------------------------------
flashcard_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
You are an expert educator. Convert the following study material into high-quality flashcards.

Each flashcard MUST follow this JSON format:

[
  {{
    "question": "...",
    "answer": "...",
    "topic": "...",
    "difficulty": "easy" | "medium" | "hard"
  }}
]

Use the text ONLY. No hallucination.
Keep answers concise.

Study Material:
{context}

Return ONLY valid JSON.
"""
)

parser = JsonOutputParser()


# -----------------------------------------------------
# Build LCEL flashcard generation chain
# -----------------------------------------------------
def build_flashcard_chain(retriever):

    # Convert retrieved docs into one text block
    def join_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    return (
        retriever
        | RunnableLambda(join_docs)
        | flashcard_prompt
        | llm
        | parser
    )


# -----------------------------------------------------
# Main function to generate flashcards
# -----------------------------------------------------
def generate_flashcards(file_path: str, query="generate flashcards"):
    print("Creating Vector Database...")
    retriever = create_vector_db(file_path)

    print("Building Flashcard Generator (LCEL + Gemini)...")
    chain = build_flashcard_chain(retriever)

    print("Generating Flashcards...")
    result = chain.invoke(query)

    return result


# -----------------------------------------------------
# Run example
# -----------------------------------------------------
if __name__ == "__main__":
    flashcards = generate_flashcards("SDE-Interview-and-Prep-Roadmap.pdf")

    print("\n Flashcards Generated:\n")
    for card in flashcards:
        print(card)
