#!/usr/bin/env -S .venv/bin/python
"""
agent_chat.py
CLI chat that searches Azure AI Search (baseline or RAPTOR index), shows retrieved context, and answers via Azure OpenAI chat.
Usage:
  ./agent_chat.py             # default index imf_baseline
  ./agent_chat.py imf_raptor  # use RAPTOR index
  DOC_ID_FILTER=IMF_2410 ./agent_chat.py imf_raptor   # optional doc filter
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from typing import List

try:
    from agent_framework import ChatAgent  # real package if present
    from agent_framework.azure import AzureOpenAIChatClient
    _USING_SHIM = False
except ImportError:
    from agent_framework_shim import ChatAgent
    from agent_framework_shim.azure import AzureOpenAIChatClient
    _USING_SHIM = True


def load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            val = val.strip().strip('"').strip("'")
            os.environ[key.strip()] = val


def http_post_json(url: str, headers: dict, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from None


def embed(texts: List[str], *, endpoint: str, deploy: str, api_key: str, api_version: str) -> List[List[float]]:
    url = f"{endpoint}/openai/deployments/{deploy}/embeddings?api-version={urllib.parse.quote(api_version)}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {"input": texts, "model": deploy}
    resp = http_post_json(url, headers, payload, timeout=60)
    return [item["embedding"] for item in resp.get("data", [])]


def chat_complete(messages: List[dict], *, endpoint: str, deployment: str, api_key: str, api_version: str) -> str:
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={urllib.parse.quote(api_version)}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {
        "messages": messages,
        "model": deployment,
        "max_completion_tokens": 240,
    }
    resp = http_post_json(url, headers, payload, timeout=60)
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in response: {resp}")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"No content in response: {resp}")
    return content


async def chat_loop() -> None:
    load_env()
    embed_deploy = os.getenv("AZURE_TEXT_EMBEDDING_DEPLOYMENT_NAME")
    chat_deploy = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    aoai_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    aoai_key = os.getenv("AZURE_OPENAI_API_KEY")
    aoai_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY") or os.getenv("AZURE_SEARCH_KEY")
    index_name = os.getenv("RAPTOR_INDEX") or os.getenv("AZURE_SEARCH_INDEX_NAME") or "imf_baseline"
    doc_filter = os.getenv("DOC_ID_FILTER")
    if len(sys.argv) > 1:
        index_name = sys.argv[1]
    if len(sys.argv) > 2:
        doc_filter = sys.argv[2]

    if not all([embed_deploy, chat_deploy, aoai_endpoint, aoai_key, search_endpoint, search_key]):
        sys.exit("❌ Missing Azure env vars. Check .env.")

    chat_client = AzureOpenAIChatClient(
        deployment_name=chat_deploy,
        endpoint=aoai_endpoint,
        api_key=aoai_key,
        api_version=aoai_version,
    )
    agent = ChatAgent(
        name="assistant",
        chat_client=chat_client,
        instructions="You answer using the provided context excerpts. Cite sources [Source n] when relevant. If context is empty, say you have no evidence.",
    )
    thread = agent.get_new_thread()
    if _USING_SHIM:
        print("ℹ️  Using local Agent Framework shim (install agent-framework to use the official package).")
    print(f"🤖 Agent chat on index '{index_name}'", end="")
    if doc_filter:
        print(f" with doc_id filter '{doc_filter}'", end="")
    print(" (type 'exit' to quit)")

    while True:
        try:
            question = input("🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        emb = embed([question], endpoint=aoai_endpoint, deploy=embed_deploy, api_key=aoai_key, api_version=aoai_version)[0]
        # Ask for common RAPTOR fields when present; fallback handled below.
        select_fields = ["id", "raw", "level", "kind", "doc_id"]
        base_filters = []
        if doc_filter:
            base_filters.append(f"doc_id eq '{doc_filter}'")

        def run_search(extra_filter: str | None, k: int, top: int) -> list[dict]:
            filters = list(base_filters)
            if extra_filter:
                filters.append(extra_filter)
            vector_payload = {
                "vectorQueries": [
                    {"kind": "vector", "vector": emb, "fields": "contentVector", "k": k}
                ],
                "select": ",".join(select_fields),
                "top": top,
                "search": question,  # hybrid: text + vector
            }
            if filters:
                vector_payload["filter"] = " and ".join(filters)
            try:
                return http_post_json(
                    f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2024-07-01",
                    {"Content-Type": "application/json", "api-key": search_key},
                    vector_payload,
                    timeout=30,
                ).get("value", [])
            except RuntimeError as e:
                msg = str(e)
                if "Could not find a property named 'doc_id'" in msg or "Could not find a property named 'level'" in msg:
                    # Fallback for indexes without RAPTOR fields
                    vector_payload.pop("filter", None)
                    vector_payload["select"] = "id,raw"
                    print("⚠️  Some fields missing in index; retrying search with id/raw only.")
                    return http_post_json(
                        f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2024-07-01",
                        {"Content-Type": "application/json", "api-key": search_key},
                        vector_payload,
                        timeout=30,
                    ).get("value", [])
                raise

        # For RAPTOR indexes, pull both summaries and leaves and merge (summaries first).
        if "raptor" in index_name.lower():
            summary_hits = run_search("level gt 0", k=5, top=5)
            leaf_hits = run_search(None, k=5, top=5)
            seen = set()
            hits = []
            for h in summary_hits + leaf_hits:
                hid = h.get("id")
                if hid in seen:
                    continue
                seen.add(hid)
                hits.append(h)
        else:
            hits = run_search(None, k=10, top=10)
        context_block = ""
        if not hits:
            context_block = "No relevant excerpts found."
            print("\n📄 Retrieved context: (none)\n")
        else:
            parts = []
            for i, h in enumerate(hits, 1):
                raw = h.get("raw", "")
                snippet = textwrap.shorten(raw, 400) if raw else "(no raw text)"
                prefix_bits = [f"id={h.get('id','')}"]
                if "level" in h:
                    prefix_bits.append(f"level={h.get('level')}")
                if "kind" in h:
                    prefix_bits.append(f"kind={h.get('kind')}")
                if "doc_id" in h:
                    prefix_bits.append(f"doc_id={h.get('doc_id')}")
                prefix = " ".join(prefix_bits)
                parts.append(f"[Source {i}] {prefix} \n{snippet}")
            context_block = "\n\n".join(parts)
            print("\n📄 Retrieved context:\n")
            print(context_block)
            print()

        prompt = (
            "Use the provided excerpts to answer.\n"
            f"{context_block}\n\n"
            f"Question: {question}\n"
            "Answer concisely and cite [Source n]."
        )

        result = await agent.run(prompt, thread=thread)
        print(f"🤖 {result.text}\n")


if __name__ == "__main__":
    asyncio.run(chat_loop())
