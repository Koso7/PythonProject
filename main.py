import os
import textwrap

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

QDRANT_DIR = os.getenv("QDRANT_DIR", "./qdrant_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-nemo-instruct-2407")
LM_STUDIO_URL = "http://localhost:1234/v1"


def format_docs(docs):
    parts = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unbekannte Quelle")
        parts.append(f"Quelle {index}: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def main():
    print("=" * 70)
    print("🏥 Pflege-KI Konsolentest (Qdrant & LM Studio)")
    print("Tippe 'exit' zum Beenden.")
    print("=" * 70)

    # Einheitliche Embeddings via LangChain OpenAI (kompatibel mit LM Studio)
    embeddings = OpenAIEmbeddings(
        openai_api_base=LM_STUDIO_URL,
        openai_api_key="lm-studio",
        model=EMBEDDING_MODEL,
        check_embedding_ctx_length=False
    )

    client = QdrantClient(path=QDRANT_DIR)
    vector_db = QdrantVectorStore(
        client=client,
        collection_name="pflege_fachwissen",
        embedding=embeddings
    )

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    )

    # Einheitliches LLM via LangChain OpenAI (kompatibel mit LM Studio)
    llm = ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
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
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template)
    ])

    chain = prompt | llm | StrOutputParser()

    while True:
        user_question = input("\nDeine Frage: ")

        if user_question.lower().strip() == "exit":
            print("Programm beendet.")
            break

        print("\n⏳ Suche relevante Fachquellen...")
        docs = retriever.invoke(user_question)
        context = format_docs(docs)

        print("⏳ KI erstellt Antwort...\n")
        response = chain.invoke({
            "context": context,
            "question": user_question,
        })

        print("🤖 Antwort:")
        print(textwrap.fill(response, width=100))

        print("\n📚 Verwendete Quellen:")
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unbekannte Quelle")
            print(f"{index}. {source}")

        print("-" * 70)


if __name__ == "__main__":
    main()