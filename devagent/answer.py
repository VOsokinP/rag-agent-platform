"""Turning retrieved chunks into a grounded, cited answer."""

from dataclasses import dataclass

from devagent.llm.provider import Provider
from devagent.retrieval.vector_search import RetrievedChunk

SYSTEM_PROMPT = """You are a coding assistant answering questions about a specific \
codebase. Answer only from the context provided below. Each context block is labelled \
with the file path and symbol it came from; refer to those labels in your answer so the \
reader can verify it. If the context does not contain enough information to answer, say \
so plainly instead of guessing."""


class EmptyCorpusError(RuntimeError):
    """Raised when there is nothing retrieved to ground an answer in."""


@dataclass(frozen=True)
class Citation:
    """Where a piece of the answer came from."""

    file_path: str
    symbol: str | None
    start_line: int
    end_line: int
    score: float


@dataclass(frozen=True)
class Answer:
    """A grounded answer and the chunks it was grounded in."""

    answer: str
    citations: list[Citation]


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as labelled blocks for the prompt."""
    blocks = []
    for chunk in chunks:
        label = chunk.symbol or "<file>"
        blocks.append(
            f"--- {chunk.file_path} :: {label} "
            f"(lines {chunk.start_line}-{chunk.end_line}) ---\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def answer_question(
    question: str, chunks: list[RetrievedChunk], provider: Provider
) -> Answer:
    """Answer a question from retrieved chunks, returning the answer and citations."""
    if not chunks:
        raise EmptyCorpusError(
            "No chunks retrieved. Has the repository been ingested yet?"
        )

    user_prompt = f"Context:\n\n{build_context(chunks)}\n\nQuestion: {question}"
    reply = provider.complete(SYSTEM_PROMPT, user_prompt)

    return Answer(
        answer=reply,
        citations=[
            Citation(
                file_path=chunk.file_path,
                symbol=chunk.symbol,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                score=chunk.score,
            )
            for chunk in chunks
        ],
    )
