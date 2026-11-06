from types import SimpleNamespace

import pytest

import main_agent
from agents import coder_agent, critic_agent, doc_agent, router
from agents.utils import strip_think


class FakeLLM:
    def __init__(self, *risposte):
        self.risposte = list(risposte)

    def invoke(self, _messaggi):
        return SimpleNamespace(content=self.risposte.pop(0))


def test_strip_think():
    assert strip_think("<think>a\nb</think>\n risposta") == "risposta"
    assert strip_think("<think>non chiuso") == ""


def test_router_ignora_il_ragionamento():
    llm = FakeLLM("<think>non doc_agent ma coder_agent</think>\ncoder_agent")
    assert router.route_query({"user_input": "scrivi"}, llm)["routed_to"] == "coder_agent"


def test_router_ha_un_default():
    assert router.route_query({"user_input": "x"}, FakeLLM("boh"))["routed_to"] == "doc_agent"


@pytest.mark.parametrize("testo,atteso", [
    ("**VALUTAZIONE QUALITÀ:** [9/10]", 9.0),
    ("VALUTAZIONE QUALITÀ: 8,5/10", 8.5),
    ("VALUTAZIONE QUALITÀ: 80/10", None),
    ("nessun punteggio", None),
])
def test_punteggio_critico(testo, atteso):
    assert critic_agent._extract_quality_score(testo) == atteso


@pytest.mark.parametrize("codice", [
    "import os",
    "from subprocess import run",
    "open('x.txt')",
    "__import__('os')",
    "().__class__",
    "import pandas as pd\npd.read_csv('x.csv')",
    "import pandas as pd\npd.DataFrame().to_csv('x.csv')",
])
def test_operazioni_pericolose_bloccate(codice):
    uscita, errore = coder_agent.execute_python_code(codice)
    assert uscita == ""
    assert "Esecuzione bloccata" in errore


def test_codice_normale_con_classi_e_min():
    codice = "class A:\n    pass\nprint(min(1, 5), isinstance(A(), A))"
    assert coder_agent.execute_python_code(codice) == ("1 True\n", "")


def test_timeout(monkeypatch):
    monkeypatch.setattr(coder_agent, "EXECUTION_TIMEOUT_SECONDS", 1)
    _, errore = coder_agent.execute_python_code("while True:\n    pass")
    assert "interrotta" in errore


def test_il_codice_non_viene_eseguito_senza_autorizzazione():
    llm = FakeLLM("```python\nprint('eseguito')\n```")
    stato = coder_agent.run_coder_agent({"user_input": "x"}, llm)
    assert stato["execution_results"] == []


def test_il_codice_nel_ragionamento_non_viene_eseguito():
    llm = FakeLLM("<think>```python\nprint('segreto')\n```</think>```python\nprint('ok')\n```")
    stato = coder_agent.run_coder_agent({"user_input": "x", "allow_code_execution": True}, llm)
    assert [r["output"] for r in stato["execution_results"]] == ["ok\n"]


def test_impronta_cambia_con_i_documenti(tmp_path):
    prima = doc_agent.documents_fingerprint(str(tmp_path))
    (tmp_path / "a.txt").write_text("ciao", encoding="utf-8")
    assert doc_agent.documents_fingerprint(str(tmp_path)) != prima


def test_grafo_completo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    risposte = iter(["<think>doc_agent?</think>coder_agent", "```python\nprint(2 + 2)\n```",
                     "**RISPOSTA VALIDATA:** quattro\n**VALUTAZIONE QUALITÀ:** [9/10]"])

    class LLM:
        def invoke(self, _messaggi):
            return SimpleNamespace(content=next(risposte))

    monkeypatch.setattr(main_agent, "create_llm", lambda **_: LLM())
    testo = main_agent.run_query(main_agent.build_graph(), "somma", allow_code_execution=True)
    assert "coder_agent → critic_agent" in testo
    assert "9.0/10" in testo
