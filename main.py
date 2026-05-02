# 1. Verbindung zur Datenbank und den Modellen
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="mistral")

# 2. Der System-Prompt
template = """Du bist ein Experte für das Neue Begutachtungsassessment (NBA) in der Pflege. 
Nutze ausschließlich die folgenden Auszüge aus dem Fachdokument, um die Frage des Nutzers zu beantworten.
Wenn du die Antwort nicht im Text findest oder der Text absolut nichts mit der Frage zu tun hat, 
sage exakt: "Dazu gibt es in meinen vorliegenden Pflege-Dokumenten keine Informationen." Erfinde nichts dazu.

Kontext: {context}

Frage: {question}

Antwort:"""

prompt = PromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 3. Die Pipeline
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# 4. Das aufgeräumte Chat-Interface
print("=" * 60)
print(" 🏥 Pflege-KI (NBA) ist bereit! (Tippe 'exit' zum Beenden)")
print("=" * 60)

while True:
    user_input = input("\nDeine Frage zum NBA: ")
    if user_input.lower() == 'exit':
        break

    print("⏳ KI überlegt...\n")

    # Die Anfrage wird direkt durch die Pipeline geschickt (ohne Quellen-Print)
    response = rag_chain.invoke(user_input)

    print("🤖 ANTWORT:")
    # Die Antwort der KI brechen wir schön nach 80 Zeichen um
    print(textwrap.fill(response, width=80))
    print("-" * 60)