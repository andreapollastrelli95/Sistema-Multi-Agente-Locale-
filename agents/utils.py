"""Funzioni condivise dagli agenti."""
import re


def strip_think(text: str) -> str:
    """Rimuove il ragionamento <think> di DeepSeek-R1, anche se il tag non viene chiuso."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
