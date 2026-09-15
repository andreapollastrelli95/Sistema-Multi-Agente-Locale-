🤖 Sistema Multi-Agente Locale — LangGraph + Ollama

Sistema multi-agente completamente locale basato su **LangGraph**, **Ollama** e **ChromaDB**.  
Nessuna API esterna richiesta — tutto gira in locale

---

## 📂 Struttura del Progetto

```
.
├── main_agent.py          # Entry point + grafo LangGraph + CLI interattiva
├── requirements.txt       # Dipendenze Python
├── agents/
│   ├── __init__.py
│   ├── router.py          # Agente router: smista le query allo specialista corretto
│   ├── doc_agent.py       # Agente documenti: RAG con ChromaDB + Ollama embeddings
│   ├── coder_agent.py     # Agente coder: generazione ed esecuzione codice Python
│   └── critic_agent.py    # Agente critico: validazione e sintesi della risposta
├── documents/             # (auto-creata) Cartella per i tuoi documenti locali
└── chroma_db/             # (auto-creata) Vector store persistente ChromaDB
```

---

## 🔄 Flusso di Esecuzione

```
[Input Utente]
      │
      ▼
  🔀 Router          ← analizza e smista la richiesta
      │
      ├──▶ 📄 Doc Agent    (domande su documenti/knowledge base)
      ├──▶ 💻 Coder Agent  (codice Python, data analysis)
      └──▶ 🔍 Critic Agent (direttamente per sintesi/validazione)
                │
                ▼
          🔍 Critic Agent  ← valida, de-allucinates, sintetizza
                │
                ▼
         [Output Finale]
```

---

## ⚙️ Prerequisiti

### 1. Installa Ollama
Scarica e installa [Ollama](https://ollama.ai) per il tuo sistema operativo.

### 2. Scarica i modelli necessari
```bash
# Modello principale per ragionamento e routing
ollama pull deepseek-r1:8b

# Modello per gli embeddings (ricerca semantica documenti)
ollama pull nomic-embed-text
```

### 3. Avvia il server Ollama
```bash
ollama serve
```

---

## 🚀 Installazione ed Esecuzione

```bash
# 1. Installa le dipendenze Python
pip install -r requirements.txt

# 2. Avvia il sistema multi-agente
python main_agent.py
```

---

## 📄 Aggiungere Documenti alla Knowledge Base

Inserisci i tuoi file nella cartella `documents/` prima di avviare il sistema.  
Formati supportati: `.txt`, `.pdf`

```
documents/
├── manuale_tecnico.pdf
├── note_progetto.txt
└── report_analisi.pdf
```

Al primo avvio con documenti disponibili, ChromaDB verrà popolato automaticamente.  
Gli indici vengono salvati in `chroma_db/` e riutilizzati nelle sessioni successive.

---

## 💡 Esempi di Utilizzo

```
👤 Tu: Scrivi uno script Python per analizzare un CSV con pandas e visualizzare le statistiche
→ Instradato a: coder_agent

👤 Tu: Cosa dice il documento sul contratto del fornitore?
→ Instradato a: doc_agent

👤 Tu: Riassumi e valida questa spiegazione: [testo]
→ Instradato a: critic_agent
```

---

## 🛠️ Comandi CLI

| Comando   | Descrizione                          |
|-----------|--------------------------------------|
| `exit`    | Chiude il sistema                    |
| `quit`    | Chiude il sistema                    |
| `clear`   | Pulisce lo schermo                   |
| `history` | Mostra lo storico della sessione     |

---

## 🔧 Configurazione

Modifica le costanti in cima a `main_agent.py` per personalizzare:

```python
REASONING_MODEL = "deepseek-r1:8b"     # Modello Ollama da usare
OLLAMA_BASE_URL = "http://localhost:11434"  # URL del server Ollama
```

Modifica `agents/doc_agent.py` per cambiare il modello di embedding:

```python
EMBEDDING_MODEL = "nomic-embed-text"   # Modello embedding Ollama
CHUNK_SIZE = 1000                       # Dimensione chunk documenti
CHUNK_OVERLAP = 200                     # Overlap tra chunk
```

---

## 📦 Dipendenze

| Libreria             | Utilizzo                                  |
|----------------------|-------------------------------------------|
| `langgraph`          | Orchestrazione del grafo multi-agente     |
| `langchain`          | Framework base per gli agenti LLM         |
| `langchain-community`| Loader documenti, vector stores           |
| `langchain-ollama`   | Integrazione con Ollama LLM               |
| `chromadb`           | Vector store locale per RAG               |
| `ollama`             | Client Python per Ollama                  |
| `pydantic`           | Validazione dati e modelli                |
| `pandas`             | Data analysis nell'agente coder           |
