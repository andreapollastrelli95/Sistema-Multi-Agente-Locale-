"""
router.py - Agente Router
Analizza l'input utente e smista il compito allo specialista corretto.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from typing import Literal


ROUTER_SYSTEM_PROMPT = """Sei un agente router intelligente. Il tuo unico compito è analizzare la richiesta dell'utente e decidere quale agente specialista deve gestirla.

Agenti disponibili:
- "doc_agent": Per domande su documenti, ricerche semantiche, RAG, analisi di testi, PDF, knowledge base.
  Usa doc_agent ANCHE per:
  * Domande concettuali o teoriche (es. "Cos'è il RAG?", "RAG vs Cloud", "Come funziona X?").
  * Spiegazioni generali, confronti tra tecnologie, definizioni e overview.
  * Qualsiasi domanda che non richieda di scrivere codice né di validare un testo già prodotto.
- "coder_agent": Per scrittura di codice Python, analisi dati, script, debugging, data analysis con pandas/numpy.
  Usa coder_agent SOLO se l'utente chiede esplicitamente di scrivere, correggere o eseguire del codice.
- "critic_agent": Per validazione di risposte, sintesi di informazioni, verifica di fatti, riassunti.
  Usa critic_agent SOLO se l'utente fornisce già un testo da rivedere o validare.

Regola di default: se hai dubbi, scegli doc_agent.

Rispondi SOLO con una delle seguenti parole: doc_agent, coder_agent, critic_agent.
Non aggiungere spiegazioni, solo la parola chiave dell'agente corretto.
"""


class RoutingDecision(BaseModel):
    agent: Literal["doc_agent", "coder_agent", "critic_agent"]
    confidence: float
    reasoning: str


def route_query(state: dict, llm: ChatOllama) -> dict:
    """
    Analizza la query utente e determina l'agente appropriato.
    
    Args:
        state: Stato condiviso del sistema multi-agente
        llm: Istanza del modello LLM Ollama
    
    Returns:
        Stato aggiornato con la decisione di routing
    """
    user_query = state.get("user_input", "")
    
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Richiesta utente: {user_query}\n\nQuale agente deve gestire questa richiesta?")
    ]
    
    response = llm.invoke(messages)
    raw_decision = response.content.strip().lower()
    
    # Normalizza la risposta
    agent_map = {
        "doc_agent": "doc_agent",
        "coder_agent": "coder_agent", 
        "critic_agent": "critic_agent",
    }
    
    routed_agent = "doc_agent"  # default fallback: domande generali → doc_agent
    for key, value in agent_map.items():
        if key in raw_decision:
            routed_agent = value
            break
    
    print(f"\n[ROUTER] Query: '{user_query[:60]}...' → Instradato a: {routed_agent}")
    
    return {
        **state,
        "routed_to": routed_agent,
        "routing_reasoning": raw_decision,
    }
