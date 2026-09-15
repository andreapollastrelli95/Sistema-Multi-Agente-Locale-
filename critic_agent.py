"""
critic_agent.py - Agente Critico/Validatore
Controlla l'output degli altri agenti, elimina le allucinazioni e sintetizza la risposta finale.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


CRITIC_SYSTEM_PROMPT = """Sei un agente validatore critico e sintetizzatore esperto.
Il tuo ruolo è fondamentale per garantire la qualità delle risposte del sistema.

I tuoi compiti:
1. **Verifica accuratezza**: Controlla che la risposta sia logicamente coerente e fattualmente plausibile
2. **Elimina allucinazioni**: Identifica e rimuovi affermazioni non supportate o palesemente errate
3. **Migliora chiarezza**: Ristruttura la risposta per renderla più chiara e leggibile
4. **Sintetizza**: Condensa le informazioni essenziali eliminando ridondanze
5. **Valuta qualità**: Assegna un punteggio di qualità da 1 a 10

Formato risposta richiesto:
---
**RISPOSTA VALIDATA:**
[Risposta migliorata e verificata]

**VALUTAZIONE QUALITÀ:** [X/10]

**NOTE CRITICA:**
[Eventuali avvertenze, limitazioni o punti da verificare]
---
"""


def run_critic_agent(state: dict, llm: ChatOllama) -> dict:
    """
    Valida e migliora l'output proveniente dagli altri agenti.
    
    Args:
        state: Stato condiviso del sistema con l'output dell'agente precedente
        llm: Istanza del modello LLM
    
    Returns:
        Stato aggiornato con la risposta validata e sintetizzata
    """
    user_query = state.get("user_input", "")
    agent_output = state.get("agent_output", "")
    agent_used = state.get("agent_used", "sconosciuto")
    retrieved_context = state.get("retrieved_context", "")
    
    print(f"[CRITIC_AGENT] Validazione output proveniente da: {agent_used}...")
    
    # Costruisci il prompt di validazione con tutto il contesto disponibile
    validation_context = f"""
**RICHIESTA ORIGINALE DELL'UTENTE:**
{user_query}

**AGENTE CHE HA GENERATO LA RISPOSTA:** {agent_used}

**RISPOSTA DA VALIDARE:**
{agent_output}
"""
    
    if retrieved_context:
        validation_context += f"""
**CONTESTO DI RIFERIMENTO (da knowledge base):**
{retrieved_context[:2000]}...
"""
    
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=f"""Analizza e valida la seguente risposta:

{validation_context}

Fornisci una versione migliorata, accurata e ben strutturata della risposta.""")
    ]
    
    response = llm.invoke(messages)
    validated_output = response.content
    
    # Estrai il punteggio di qualità se presente
    quality_score = _extract_quality_score(validated_output)
    
    print(f"[CRITIC_AGENT] Validazione completata. Qualità: {quality_score}/10")
    
    return {
        **state,
        "final_output": validated_output,
        "quality_score": quality_score,
        "agent_used": f"{agent_used} → critic_agent",
    }


def _extract_quality_score(text: str) -> float:
    """
    Estrae il punteggio di qualità dalla risposta del critico.
    
    Args:
        text: Testo della risposta del critico
    
    Returns:
        Punteggio numerico (default 7.0 se non trovato)
    """
    import re
    pattern = r"VALUTAZIONE QUALITÀ[:\s]*(\d+(?:\.\d+)?)\s*/\s*10"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 7.0
