"""
doc_agent.py - Agente Documenti (RAG)
Dedicato alla ricerca semantica/ibrida su documenti locali tramite ChromaDB e Ollama embeddings.
"""

import os
from typing import Optional
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOC_AGENT_SYSTEM_PROMPT = """Sei un agente specializzato nell'analisi e ricerca di documenti.
Hai accesso a una knowledge base vettoriale. Usa il contesto recuperato per rispondere 
in modo preciso e citando le fonti quando disponibili.

Se il contesto recuperato non contiene informazioni rilevanti, comunicalo chiaramente.
Rispondi in modo strutturato, dettagliato e accurato.
"""

# Directory predefinita per i documenti locali
DOCS_DIR = "./documents"
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"  # Modello embedding locale Ollama
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def build_vector_store(docs_directory: str = DOCS_DIR) -> Optional[Chroma]:
    """
    Costruisce o carica il vector store ChromaDB dai documenti locali.
    
    Args:
        docs_directory: Percorso alla cartella dei documenti
    
    Returns:
        Istanza ChromaDB o None se non ci sono documenti
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Carica il vector store esistente se disponibile
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"[DOC_AGENT] Caricamento vector store esistente da '{CHROMA_PERSIST_DIR}'...")
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
    
    # Controlla se esiste la directory dei documenti
    if not os.path.exists(docs_directory):
        print(f"[DOC_AGENT] Directory documenti '{docs_directory}' non trovata. Creazione...")
        os.makedirs(docs_directory, exist_ok=True)
        return None
    
    # Carica i documenti disponibili
    documents = []
    loaders = [
        DirectoryLoader(docs_directory, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True),
        DirectoryLoader(docs_directory, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True),
    ]
    
    for loader in loaders:
        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"[DOC_AGENT] Errore caricamento documenti: {e}")
    
    if not documents:
        print("[DOC_AGENT] Nessun documento trovato nella knowledge base.")
        return None
    
    # Suddivisione in chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    
    print(f"[DOC_AGENT] Indicizzati {len(chunks)} chunk da {len(documents)} documenti.")
    
    # Crea e persiste il vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    
    return vector_store


def retrieve_context(query: str, vector_store: Chroma, k: int = 4) -> str:
    """
    Recupera il contesto rilevante tramite ricerca semantica.
    
    Args:
        query: Query di ricerca
        vector_store: Vector store ChromaDB
        k: Numero di chunk da recuperare
    
    Returns:
        Contesto formattato come stringa
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    
    if not results:
        return "Nessun documento rilevante trovato nella knowledge base."
    
    context_parts = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Sconosciuta")
        context_parts.append(
            f"[Fonte {i} - Score: {score:.3f} | File: {os.path.basename(source)}]\n{doc.page_content}"
        )
    
    return "\n\n---\n\n".join(context_parts)


def run_doc_agent(state: dict, llm: ChatOllama) -> dict:
    """
    Esegue l'agente documenti: recupera contesto e genera risposta.
    
    Args:
        state: Stato condiviso del sistema
        llm: Istanza del modello LLM
    
    Returns:
        Stato aggiornato con la risposta dell'agente documenti
    """
    user_query = state.get("user_input", "")
    
    print("[DOC_AGENT] Avvio ricerca semantica nella knowledge base...")
    
    # Carica o costruisce il vector store
    vector_store = build_vector_store()
    
    if vector_store is None:
        context = "Knowledge base vuota. Aggiungi documenti nella cartella './documents' e riavvia."
    else:
        context = retrieve_context(user_query, vector_store)
    
    print(f"[DOC_AGENT] Contesto recuperato ({len(context)} caratteri). Generazione risposta...")
    
    messages = [
        SystemMessage(content=DOC_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""Contesto dalla knowledge base:
{context}

---

Domanda dell'utente: {user_query}

Fornisci una risposta basata sul contesto recuperato.""")
    ]
    
    response = llm.invoke(messages)
    
    print("[DOC_AGENT] Risposta generata.")
    
    return {
        **state,
        "agent_output": response.content,
        "agent_used": "doc_agent",
        "retrieved_context": context,
    }
