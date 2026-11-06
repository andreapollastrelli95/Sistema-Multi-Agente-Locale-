"""
main_agent.py - Sistema Multi-Agente con LangGraph
Entry point principale che configura il grafo di esecuzione e la CLI interattiva.
"""

import argparse
import os
from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from agents.router import route_query
from agents.doc_agent import run_doc_agent
from agents.coder_agent import run_coder_agent
from agents.critic_agent import run_critic_agent


# ─────────────────────────────────────────────
# CONFIGURAZIONE MODELLI OLLAMA
# ─────────────────────────────────────────────

REASONING_MODEL = "deepseek-r1:8b"   # Modello principale per routing e ragionamento
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🤖  SISTEMA MULTI-AGENTE LOCALE  🤖                ║
║                  Powered by LangGraph + Ollama               ║
╠══════════════════════════════════════════════════════════════╣
║  Agenti disponibili:                                         ║
║   🔀 Router       → Smistamento intelligente delle query     ║
║   📄 Doc Agent    → RAG e ricerca semantica su documenti     ║
║   💻 Coder Agent  → Scrittura codice Python                  ║
║   🔍 Critic Agent → Validazione e sintesi della risposta     ║
╠══════════════════════════════════════════════════════════════╣
║  Modello: {model:<51}║
║  Comandi: 'exit' per uscire | 'clear' per pulire schermo     ║
╚══════════════════════════════════════════════════════════════╝
""".format(model=REASONING_MODEL)


# ─────────────────────────────────────────────
# STATO CONDIVISO (SharedState)
# ─────────────────────────────────────────────

class SharedState(TypedDict):
    """Stato condiviso tra tutti gli agenti nel grafo LangGraph."""
    user_input: str                    # Input originale dell'utente
    allow_code_execution: bool         # Il coder puo' eseguire il codice generato
    routed_to: Optional[str]          # Agente di destinazione scelto dal router
    routing_reasoning: Optional[str]  # Motivazione del routing
    agent_output: Optional[str]       # Output dell'agente specialista
    agent_used: Optional[str]         # Nome dell'agente che ha risposto
    retrieved_context: Optional[str]  # Contesto recuperato dal doc_agent
    execution_results: Optional[Any]  # Risultati esecuzione codice (coder_agent)
    final_output: Optional[str]       # Output finale validato dal critic
    quality_score: Optional[float]    # Punteggio qualità assegnato dal critic
    error: Optional[str]              # Eventuali errori durante l'elaborazione


# ─────────────────────────────────────────────
# INIZIALIZZAZIONE LLM
# ─────────────────────────────────────────────

def create_llm(model: str = REASONING_MODEL, temperature: float = 0.1) -> ChatOllama:
    """
    Crea un'istanza del modello LLM Ollama.
    
    Args:
        model: Nome del modello Ollama da utilizzare
        temperature: Temperatura per la generazione (0=deterministico, 1=creativo)
    
    Returns:
        Istanza ChatOllama configurata
    """
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


# ─────────────────────────────────────────────
# NODI DEL GRAFO LANGGRAPH
# ─────────────────────────────────────────────

def node_router(state: SharedState) -> SharedState:
    """Nodo: Router - Analizza e smista la query."""
    llm = create_llm(temperature=0.0)
    return route_query(state, llm)


def node_doc_agent(state: SharedState) -> SharedState:
    """Nodo: Doc Agent - RAG e ricerca semantica."""
    llm = create_llm(temperature=0.1)
    return run_doc_agent(state, llm)


def node_coder_agent(state: SharedState) -> SharedState:
    """Nodo: Coder Agent - Generazione ed esecuzione codice."""
    llm = create_llm(temperature=0.2)
    return run_coder_agent(state, llm)


def node_critic_agent(state: SharedState) -> SharedState:
    """Nodo: Critic Agent - Validazione e sintesi finale."""
    llm = create_llm(temperature=0.1)
    return run_critic_agent(state, llm)


# ─────────────────────────────────────────────
# LOGICA DI ROUTING CONDIZIONALE
# ─────────────────────────────────────────────

def should_route_to(state: SharedState) -> str:
    """
    Determina il prossimo nodo del grafo in base alla decisione del router.
    Il critic_agent NON è raggiungibile direttamente dal router: è sempre
    un passo successivo a doc_agent o coder_agent.
    
    Args:
        state: Stato corrente del sistema
    
    Returns:
        Nome del nodo successivo da eseguire
    """
    routed_to = state.get("routed_to", "doc_agent")

    # critic_agent non deve mai essere il primo nodo dopo il router:
    # se il router lo sceglie (es. per domande concettuali ambigue),
    # lo rimappiamo a doc_agent che gestisce anche le risposte generali.
    if routed_to == "critic_agent":
        print("[GRAPH] critic_agent non raggiungibile direttamente dal router → rimappato a doc_agent")
        return "doc_agent"

    valid_routes = {"doc_agent", "coder_agent"}
    if routed_to not in valid_routes:
        print(f"[GRAPH] Route '{routed_to}' non valida, fallback a doc_agent")
        return "doc_agent"

    return routed_to


def should_run_critic(state: SharedState) -> str:
    """
    Dopo che uno specialista ha girato, verifica se ha prodotto output.
    Se agent_output è vuoto o assente il grafo termina senza invocare il critic,
    evitando una validazione su contenuto inesistente.

    Args:
        state: Stato corrente del sistema

    Returns:
        'critic_agent' se l'output esiste, 'end_no_output' altrimenti
    """
    agent_output = state.get("agent_output", "") or ""
    if agent_output.strip():
        return "critic_agent"

    print("[GRAPH] agent_output vuoto: il critic_agent viene saltato.")
    return "end_no_output"


def node_end_no_output(state: SharedState) -> SharedState:
    """Nodo terminale di fallback: nessun agente ha prodotto output."""
    return {
        **state,
        "final_output": "[⚠️ Sistema] Nessun output generato dall'agente specialista. Riprova con una domanda diversa.",
        "quality_score": 0.0,
    }


# ─────────────────────────────────────────────
# COSTRUZIONE DEL GRAFO LANGGRAPH
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Costruisce e compila il grafo di esecuzione LangGraph.

    Flusso del grafo:
      [START] → router → {doc_agent | coder_agent}
                             │
                    [has agent_output?]
                       ├─ Sì → critic_agent → [END]
                       └─ No → end_no_output → [END]

    Il critic_agent è SEMPRE preceduto da uno specialista
    e viene eseguito SOLO se agent_output non è vuoto.

    Returns:
        Grafo LangGraph compilato e pronto all'uso
    """
    graph = StateGraph(SharedState)

    # Nodi specialisti e di supporto
    graph.add_node("router", node_router)
    graph.add_node("doc_agent", node_doc_agent)
    graph.add_node("coder_agent", node_coder_agent)
    graph.add_node("critic_agent", node_critic_agent)
    graph.add_node("end_no_output", node_end_no_output)

    # Entry point
    graph.set_entry_point("router")

    # Edge condizionale: router → {doc_agent | coder_agent}
    # (critic_agent non è mai raggiungibile direttamente)
    graph.add_conditional_edges(
        "router",
        should_route_to,
        {
            "doc_agent": "doc_agent",
            "coder_agent": "coder_agent",
        }
    )

    # Edge condizionale dopo ogni specialista:
    # → critic_agent  se agent_output non è vuoto
    # → end_no_output se agent_output è vuoto
    for specialist in ("doc_agent", "coder_agent"):
        graph.add_conditional_edges(
            specialist,
            should_run_critic,
            {
                "critic_agent": "critic_agent",
                "end_no_output": "end_no_output",
            }
        )

    # Entrambi i terminali → END
    graph.add_edge("critic_agent", END)
    graph.add_edge("end_no_output", END)

    return graph.compile()


# ─────────────────────────────────────────────
# ESECUZIONE DI UNA QUERY
# ─────────────────────────────────────────────

def run_query(graph, user_input: str, allow_code_execution: bool = False) -> str:
    """
    Esegue una query attraverso il grafo multi-agente.
    
    Args:
        graph: Grafo LangGraph compilato
        user_input: Domanda/richiesta dell'utente
        allow_code_execution: Autorizza l'esecuzione del codice generato
    
    Returns:
        Risposta finale elaborata dal sistema
    """
    initial_state: SharedState = {
        "user_input": user_input,
        "allow_code_execution": allow_code_execution,
        "routed_to": None,
        "routing_reasoning": None,
        "agent_output": None,
        "agent_used": None,
        "retrieved_context": None,
        "execution_results": None,
        "final_output": None,
        "quality_score": None,
        "error": None,
    }
    
    try:
        final_state = graph.invoke(initial_state)
        
        output = final_state.get("final_output") or final_state.get("agent_output", "")
        quality = final_state.get("quality_score")
        quality_label = f"{quality}/10" if quality is not None else "n/d"
        agent = final_state.get("agent_used", "sconosciuto")
        
        # Intestazione del risultato
        separator = "═" * 64
        header = f"\n{separator}\n📊 Agente: {agent} | Autovalutazione del critico: {quality_label}\n{separator}\n"
        
        return header + output + f"\n{separator}\n"
    
    except Exception as e:
        error_msg = f"[ERRORE] Eccezione durante l'elaborazione: {e}"
        print(error_msg)
        return error_msg


# ─────────────────────────────────────────────
# INTERFACCIA CLI INTERATTIVA
# ─────────────────────────────────────────────

def cli_loop(graph, allow_code_execution: bool = False) -> None:
    """
    Avvia il loop interattivo della CLI.
    
    Args:
        graph: Grafo LangGraph compilato
        allow_code_execution: Autorizza l'esecuzione del codice generato
    """
    print(BANNER)
    print("✅ Sistema pronto. Digita la tua domanda e premi INVIO.\n")
    
    history = []  # Storico delle interazioni nella sessione
    
    while True:
        try:
            user_input = input("👤 Tu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Arrivederci!")
            break
        
        if not user_input:
            continue
        
        # Comandi speciali
        if user_input.lower() in ("exit", "quit", "esci"):
            print("\n👋 Arrivederci!")
            break
        
        if user_input.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)
            continue
        
        if user_input.lower() == "history":
            if not history:
                print("  [Nessuna interazione nella sessione corrente]\n")
            else:
                print("\n📋 Storico sessione:")
                for i, (q, _) in enumerate(history, 1):
                    print(f"  {i}. {q[:80]}...")
                print()
            continue
        
        # Elaborazione della query
        print(f"\n⏳ Elaborazione in corso...\n")
        response = run_query(graph, user_input, allow_code_execution)
        print(response)
        
        # Salva nello storico
        history.append((user_input, response))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    """Funzione principale: costruisce il grafo e avvia la CLI."""
    parser = argparse.ArgumentParser(description="Sistema multi-agente locale con LangGraph e Ollama")
    parser.add_argument(
        "--esegui-codice",
        action="store_true",
        help="Esegue sul computer il codice generato dal modello, entro i limiti di agents/coder_agent.py",
    )
    args = parser.parse_args()

    print("🔧 Inizializzazione del sistema multi-agente...")
    print(f"   Modello: {REASONING_MODEL}")
    print(f"   Backend: Ollama @ {OLLAMA_BASE_URL}")
    
    # Verifica che Ollama sia raggiungibile
    try:
        import ollama
        models = ollama.list()
        print(f"   Modelli disponibili: {len(models.get('models', []))}")
    except Exception as e:
        print(f"\n⚠️  Attenzione: impossibile connettersi a Ollama ({e})")
        print("   Assicurati che Ollama sia avviato con: ollama serve")
        print("   E che il modello sia disponibile con: ollama pull deepseek-r1:8b\n")
    
    # Costruzione del grafo LangGraph
    print("🔗 Costruzione del grafo LangGraph...")
    graph = build_graph()
    print("✅ Grafo costruito con successo.\n")
    
    # Avvio della CLI interattiva
    cli_loop(graph, args.esegui_codice)


if __name__ == "__main__":
    main()
