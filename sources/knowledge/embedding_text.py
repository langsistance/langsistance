from typing import Any


def _clean_part(value: Any) -> str:
    return " ".join(("" if value is None else str(value)).split())


def build_knowledge_embedding_text(
    question: str,
    description: str | None = "",
    answer: str | None = "",
) -> str:
    parts = []
    question_text = _clean_part(question)
    if question_text:
        parts.append(f"Question:\n{question_text}")

    description_text = _clean_part(description)
    if description_text:
        parts.append(f"Routing hint:\n{description_text}")

    answer_text = _clean_part(answer)
    if answer_text:
        parts.append(f"Knowledge content:\n{answer_text}")

    return "\n\n".join(parts)
