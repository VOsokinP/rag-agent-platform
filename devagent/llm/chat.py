"""The tool-calling chat model, kept behind one function.

`provider.py` promises the OpenAI dependency can be swapped without touching
ingestion, retrieval, or the API layer. The agent needs tool calling, which the
Provider protocol does not offer, so the model is constructed here rather than
inside the graph -- otherwise that promise would quietly stop being true.
"""

from langchain_openai import ChatOpenAI

from devagent.config import get_settings


def chat_model() -> ChatOpenAI:
    """Return the configured tool-calling chat model."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of a run from configured per-model rates.

    An estimate from static rates, not a billed figure. Named and documented
    that way so a number on a dashboard is not mistaken for an invoice.
    """
    settings = get_settings()
    return (
        prompt_tokens / 1_000_000 * settings.chat_input_cost_per_mtok
        + completion_tokens / 1_000_000 * settings.chat_output_cost_per_mtok
    )
