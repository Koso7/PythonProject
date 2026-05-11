import os
import textwrap

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-nemo")


def format_docs(docs):
    parts = []

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unbekannte Quelle")
        page = doc.metadata.get("page", None)

        if page is not None:
            source_text = f"Quelle {index}: {source}, Seite {page + 1}"
        else:
            source_text = f"Quelle {index}: {source}"

        parts.append(
            f"{source_text}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


def main():
    print("=" * 70)
    print("🏥 Pflege-KI Konsolentest")
    print("Tippe 'exit' zum Beenden.")
    print("=" * 70)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 8})

    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0.0,
    )

    template = """
Du bist ein KI-gestützter Assistenzdienst zum Thema Pflegegrad und Pflegegrad-Widerspruch.

Regeln:
- Antworte ausschließlich auf Grundlage des bereitgestellten Kontexts.
- Erfinde keine rechtlichen, medizinischen oder pflegefachlichen Fakten.
- Wenn die Antwort im Kontext nicht enthalten ist, sage:
  "Dazu gibt es in den vorliegenden Pflege-Dokumenten keine ausreichenden Informationen."
- Schreibe vollständig auf Deutsch.
- Weise bei rechtlichen Fragen darauf hin, dass keine Rechtsberatung ersetzt wird.

Kontext:
{context}

Frage:
{question}

Antwort:
"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    while True:
        user_question = input("\nDeine Frage: ")

        if user_question.lower().strip() == "exit":
            print("Programm beendet.")
            break

        print("\n⏳ Suche relevante Dokumente...")

        docs = retriever.invoke(user_question)
        context = format_docs(docs)

        print("⏳ KI erstellt Antwort...\n")

        response = chain.invoke(
            {
                "context": context,
                "question": user_question,
            }
        )

        print("🤖 Antwort:")
        print(textwrap.fill(response, width=100))

        print("\n📚 Verwendete Quellen:")
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unbekannte Quelle")
            page = doc.metadata.get("page", None)

            if page is not None:
                print(f"{index}. {source}, Seite {page + 1}")
            else:
                print(f"{index}. {source}")

        print("-" * 70)


if __name__ == "__main__":
    main()