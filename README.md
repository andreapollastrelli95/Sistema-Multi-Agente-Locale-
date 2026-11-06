# Sistema multi-agente locale

Un assistente a riga di comando che smista ogni richiesta allo specialista adatto e fa rivedere la risposta da un agente critico. Usa LangGraph, Ollama e ChromaDB, senza API esterne.

## Flusso

```text
richiesta → router ─┬─ doc_agent ───┐
                    └─ coder_agent ─┴─ critic_agent → risposta
```

- **Router:** sceglie lo specialista. Se indica `critic_agent` o una risposta non valida, la richiesta passa a `doc_agent`.
- **Doc agent:** cerca nei documenti di `documents/` con ChromaDB e `nomic-embed-text`. L'indice si ricostruisce quando i file cambiano.
- **Coder agent:** scrive codice Python e, solo se autorizzato, lo esegue.
- **Critic agent:** riscrive la risposta e le assegna un punteggio. È un'autovalutazione del modello, non una misura oggettiva.

## Avvio

```bash
pip install -r requirements.txt
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
python main_agent.py
```

`documents/` contiene un documento dimostrativo con dati fittizi; si possono aggiungere file `.txt` e `.pdf`. Nella CLI sono disponibili `exit`, `clear` e `history`.

## Esecuzione del codice

Per impostazione predefinita il codice generato viene solo mostrato. Con `python main_agent.py --esegui-codice` viene eseguito in un processo separato, in una cartella temporanea e con un limite di 30 secondi. Prima un controllo blocca gli import fuori da una lista consentita e l'accesso esplicito ai file.

**Non è una sandbox di sicurezza.** Usa questa opzione solo se comprendi il rischio, oppure esegui il programma in un container o una macchina virtuale.

## Test

```bash
pip install -r requirements-dev.txt
pytest
```

I test simulano il modello, quindi coprono la logica del grafo e degli agenti, non la qualità delle risposte di DeepSeek.

## Limiti

- Non conserva la conversazione: ogni richiesta è indipendente, e `history` mostra solo le domande della sessione.
- Il critico usa lo stesso modello che ha generato la risposta e può introdurre errori riscrivendola.
- `langchain-community`, usato per Chroma e per il caricamento dei documenti, è in fase di dismissione: in futuro andrà sostituito.
