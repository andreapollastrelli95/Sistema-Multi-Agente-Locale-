"""
coder_agent.py - Agente Coder & Data Analysis
Specializzato nella scrittura ed esecuzione di codice Python e data analysis.
"""

import sys
import io
import traceback
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


CODER_SYSTEM_PROMPT = """Sei un esperto ingegnere software e data scientist Python.
Il tuo compito è:
1. Scrivere codice Python pulito, efficiente e ben commentato
2. Analizzare dati con pandas, numpy e matplotlib
3. Debuggare codice esistente
4. Spiegare concetti di programmazione

Quando scrivi codice:
- Usa sempre blocchi ```python ... ``` per delimitare il codice
- Aggiungi commenti esplicativi
- Gestisci le eccezioni dove necessario
- Preferisci soluzioni idiomatiche Python

Se ti viene chiesto di eseguire codice, fornisci prima il codice e poi l'output atteso.
"""

SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "sum": sum,
    "min": max,
    "max": max,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "__import__": __import__,
}


def extract_code_blocks(text: str) -> list[str]:
    """
    Estrae tutti i blocchi di codice Python dal testo.
    
    Args:
        text: Testo contenente blocchi di codice markdown
    
    Returns:
        Lista di blocchi di codice estratti
    """
    code_blocks = []
    lines = text.split("\n")
    in_block = False
    current_block = []
    
    for line in lines:
        if line.strip().startswith("```python"):
            in_block = True
            current_block = []
        elif line.strip() == "```" and in_block:
            in_block = False
            if current_block:
                code_blocks.append("\n".join(current_block))
        elif in_block:
            current_block.append(line)
    
    return code_blocks


def execute_python_code(code: str) -> tuple[str, str]:
    """
    Esegue codice Python in modo sicuro e cattura stdout/stderr.
    
    Args:
        code: Codice Python da eseguire
    
    Returns:
        Tuple (output, error)
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Namespace di esecuzione con le librerie consentite
    exec_globals = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "__main__",
    }
    
    # Importa librerie di analisi dati nel namespace
    try:
        import pandas as pd
        import numpy as np
        exec_globals["pd"] = pd
        exec_globals["np"] = np
    except ImportError:
        pass
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    error_output = ""
    try:
        exec(code, exec_globals)
    except Exception:
        error_output = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    
    return stdout_output, error_output or stderr_output


def run_coder_agent(state: dict, llm: ChatOllama) -> dict:
    """
    Esegue l'agente coder: genera codice, lo esegue e restituisce i risultati.
    
    Args:
        state: Stato condiviso del sistema
        llm: Istanza del modello LLM
    
    Returns:
        Stato aggiornato con la risposta dell'agente coder
    """
    user_query = state.get("user_input", "")
    
    print("[CODER_AGENT] Analisi richiesta e generazione codice...")
    
    messages = [
        SystemMessage(content=CODER_SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]
    
    response = llm.invoke(messages)
    generated_code_response = response.content
    
    # Estrai ed esegui eventuali blocchi di codice
    code_blocks = extract_code_blocks(generated_code_response)
    execution_results = []
    
    if code_blocks:
        print(f"[CODER_AGENT] Rilevati {len(code_blocks)} blocchi di codice. Esecuzione...")
        for i, code in enumerate(code_blocks, 1):
            stdout, error = execute_python_code(code)
            result = {
                "block_index": i,
                "code": code,
                "output": stdout,
                "error": error,
            }
            execution_results.append(result)
            
            if stdout:
                print(f"[CODER_AGENT] Output blocco {i}:\n{stdout}")
            if error:
                print(f"[CODER_AGENT] Errore blocco {i}:\n{error}")
    
    # Costruisci l'output finale arricchito con i risultati di esecuzione
    final_output = generated_code_response
    if execution_results:
        final_output += "\n\n---\n**Risultati Esecuzione:**\n"
        for res in execution_results:
            if res["output"]:
                final_output += f"\n**Output (Blocco {res['block_index']}):**\n```\n{res['output']}\n```"
            if res["error"]:
                final_output += f"\n**Errore (Blocco {res['block_index']}):**\n```\n{res['error']}\n```"
    
    print("[CODER_AGENT] Risposta completata.")
    
    return {
        **state,
        "agent_output": final_output,
        "agent_used": "coder_agent",
        "execution_results": execution_results,
    }
