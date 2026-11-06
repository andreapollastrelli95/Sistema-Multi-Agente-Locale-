"""
coder_agent.py - Agente Coder & Data Analysis
Specializzato nella scrittura ed esecuzione di codice Python e data analysis.
"""

import ast
import os
import subprocess
import sys
import tempfile

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.utils import strip_think


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

Il codice puo' usare solo: pandas, numpy e librerie standard di calcolo. Non leggere o scrivere file.
"""

ALLOWED_IMPORTS = {
    "pandas", "numpy", "math", "statistics", "json", "datetime", "re",
    "collections", "itertools", "functools", "random", "decimal", "string",
}
BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "input", "breakpoint", "memoryview",
}
EXECUTION_TIMEOUT_SECONDS = 30


def validate_code(code: str) -> list[str]:
    """
    Applica una lista di operazioni consentite al codice generato dal modello.
    Riduce i rischi ma non equivale a una sandbox: per questo l'esecuzione e' disattivata di default.

    Returns:
        Elenco dei problemi trovati; vuoto se il codice puo' essere eseguito
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Errore di sintassi: {exc}"]

    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        problems.extend(
            f"Import non consentito: {module}"
            for module in modules
            if module.split(".")[0] not in ALLOWED_IMPORTS
        )
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            problems.append(f"Funzione non consentita: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            problems.append(f"Attributo non consentito: {node.attr}")
        if isinstance(node, ast.Attribute) and node.attr.startswith(("read_", "to_")):
            problems.append(f"Accesso a file non consentito: {node.attr}")
    return problems


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
    Esegue il codice in un processo separato, in una cartella temporanea e con un limite di tempo.

    Args:
        code: Codice Python da eseguire

    Returns:
        Tuple (output, error)
    """
    problems = validate_code(code)
    if problems:
        return "", "Esecuzione bloccata. " + "; ".join(problems)

    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP") if key in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory() as folder:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=folder,
                env=env,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=EXECUTION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "", f"Esecuzione interrotta dopo {EXECUTION_TIMEOUT_SECONDS} secondi."
    return result.stdout, result.stderr


def run_coder_agent(state: dict, llm: ChatOllama) -> dict:
    """
    Esegue l'agente coder: genera codice e, se autorizzato, lo esegue.
    
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
    generated_code_response = strip_think(response.content)
    
    # Estrai ed esegui eventuali blocchi di codice
    code_blocks = extract_code_blocks(generated_code_response)
    execution_results = []
    
    allow_execution = bool(state.get("allow_code_execution"))
    if code_blocks and not allow_execution:
        print("[CODER_AGENT] Esecuzione disattivata: avvia con --esegui-codice per abilitarla.")
    if code_blocks and allow_execution:
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
