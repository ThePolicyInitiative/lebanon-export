from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Tuple

import requests

from agents.llm_clients import chat_completion


def approx_tokens(text: str) -> int:
    """Rough token estimate used only for prompt trimming."""
    return len(text) // 4


def _openrouter_headers() -> dict[str, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return {}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = os.environ.get("OPENROUTER_SITE_URL", "").strip()
    app_name = os.environ.get("OPENROUTER_APP_NAME", "Lebanon Export Dashboard").strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return headers


# Optional OpenRouter web plugin used only when explicitly enabled by the caller.
def search_with_openrouter(query: str, max_results: int = 3) -> list[str]:
    headers = _openrouter_headers()
    if not headers:
        return []
    url = "https://openrouter.ai/api/v1/chat/completions"
    data = {
        "model": os.environ.get("OPENROUTER_SEARCH_MODEL", "openai/gpt-4o-mini"),
        "messages": [{"role": "user", "content": f"Search the web and summarise: {query}"}],
        "plugins": [{"id": "web", "max_results": max_results}],
        "max_tokens": 500,
        "temperature": 0.3,
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return [content] if content else []
    except Exception as exc:
        print(f"OpenRouter search error: {exc}")
        return []


def _normalise_terms(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "from", "with", "about", "what", "which", "when",
        "where", "why", "how", "does", "did", "are", "is", "was", "were",
        "lebanon", "lebanese", "export", "exports", "dashboard", "tell", "me",
        "this", "that", "these", "those", "into", "than", "then", "have", "has",
    }
    return {
        word for word in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(word) > 2 and word not in stop
    }


def _best_local_evidence(question: str, chunks: list[dict], limit: int = 4) -> list[tuple[str, str]]:
    """Extract the most relevant sentences from retrieved local material."""
    query_terms = _normalise_terms(question)
    candidates: list[tuple[float, str, str]] = []
    for chunk in chunks:
        source = str(chunk.get("source", "local source"))
        text = re.sub(r"\s+", " ", str(chunk.get("text", "")).strip())
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            if len(sentence) < 35:
                continue
            terms = _normalise_terms(sentence)
            overlap = len(query_terms & terms)
            if query_terms and overlap == 0:
                continue
            score = overlap * 3 + (overlap / max(1, len(query_terms))) + min(len(sentence), 240) / 1000
            candidates.append((score, source, sentence))
    candidates.sort(key=lambda item: item[0], reverse=True)
    output: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, source, sentence in candidates:
        key = sentence.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        output.append((source, sentence[:420]))
        if len(output) >= limit:
            break
    return output


def _offline_answer(question: str, chunks: list[dict]) -> str:
    """Produce a concise evidence synthesis when no hosted model is configured."""
    evidence = _best_local_evidence(question, chunks, limit=5)
    if not evidence:
        return (
            "**Answer**\n\n"
            "The dashboard and local project evidence do not contain enough directly matched information to answer this reliably. "
            "I will not fill the gap with unsupported assumptions."
        )

    first_source, first_sentence = evidence[0]
    lines = [
        "**Answer**",
        "",
        first_sentence,
    ]
    if len(evidence) > 1:
        lines.extend(["", "**Supporting evidence**", ""])
        for source, sentence in evidence[1:]:
            lines.append(f"- **{source}:** {sentence}")
    lines.extend([
        "",
        "**Evidence boundary**",
        "",
        "- This answer reflects the closest local evidence. It does not establish a causal conclusion unless the cited material does so explicitly.",
    ])
    return "\n".join(lines)


def answer_general(
    question: str,
    history: List[dict],
    provider: str = "groq",
    model: str = "openai/gpt-oss-20b",
    app_help: str = "",
    retriever: Optional[Callable] = None,
    top_k: int = 3,
    use_web_search: bool = False,
) -> Tuple[str, List[dict]]:
    provider = provider.lower().strip()
    if provider not in {"groq", "openrouter"}:
        return f"Provider '{provider}' is not supported in the Streamlit agent.", []

    if provider == "groq":
        has_model_key = bool(
            os.environ.get("GROQ_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
        )
    else:
        has_model_key = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())

    retrieved_chunks = []
    if retriever is not None:
        try:
            retrieved_chunks = retriever(question, k=top_k)
        except Exception:
            pass

    web_texts = []
    if use_web_search and os.environ.get("OPENROUTER_API_KEY"):
        try:
            web_texts = search_with_openrouter(question, max_results=3)
        except Exception:
            pass

    context_parts = []
    if retrieved_chunks:
        context_parts.append("Knowledge base excerpts:")
        for chunk in retrieved_chunks[:3]:
            text = chunk.get("text", "")
            source = chunk.get("source", "unknown")
            if text:
                context_parts.append(f"[Source: {source}]\n{text[:500]}")
    if web_texts:
        context_parts.append("Recent web search results:")
        for text in web_texts[:2]:
            context_parts.append(text[:800])

    context_section = "\n\n".join(context_parts)
    if approx_tokens(context_section) > 1000:
        context_section = context_section[:3000] + "\n[...truncated]"

    if not has_model_key:
        return _offline_answer(question, retrieved_chunks), retrieved_chunks

    system_prompt = (
        "You are a senior trade-policy analyst assisting users of the Lebanon Industrial Export Dashboard. "
        "Answer the exact question asked and rely only on the supplied context.\n"
        f"Context: {context_section}\n"
        f"Application scope: {app_help}\n"
        "Reasoning method: identify the requested metric and scope; state the direct result first; distinguish observed facts, calculated implications, "
        "and unsupported causal explanations; compare against the most relevant baseline when the context provides one; and surface a material limitation "
        "only when it changes the interpretation. Preserve every figure, year, product name, country name, and unit exactly. Use product names as the primary identifier and omit HS codes unless the user explicitly asks for or supplies a code. Do not introduce statistics that "
        "are absent from the context. Use compact bold labels only when they improve clarity, concise hyphen bullets, and complete sentences. Do not use "
        "H1/H2/H3 headings, tables, blockquotes, emojis, code fences, or suggested prompts. Keep ordinary answers under 400 words unless the user requests detail."
    )

    trimmed_history = history[-12:] if len(history) > 12 else history
    messages = list(trimmed_history)
    messages.append({"role": "user", "content": question})

    total_chars = len(system_prompt) + sum(len(str(message.get("content", ""))) for message in messages)
    if total_chars // 4 > 5000:
        system_prompt = system_prompt[:2000] + "\n[...]"

    try:
        answer = chat_completion(
            provider=provider,
            system_prompt=system_prompt,
            messages=messages,
            model=model,
            temperature=0.25,
            max_tokens=1000,
        )
        return answer, retrieved_chunks
    except Exception:
        return _offline_answer(question, retrieved_chunks), retrieved_chunks
