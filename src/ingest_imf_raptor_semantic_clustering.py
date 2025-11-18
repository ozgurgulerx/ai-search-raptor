#!/usr/bin/env -S .venv/bin/python
"""
ingest_raptor_semantic_clustering.py
────────────────────────────────────
Build a RAPTOR-style hierarchical index in Azure AI Search from IMF outlook text
stored under data/ (e.g., data/IMF_2510.txt), using semantic clustering (greedy
nearest-neighbor grouping in embedding space) at each level instead of fixed
contiguous grouping.

Schema (same as ingest_raptor.py):
  - id (key), level (filterable/facetable), kind ("chunk"|"summary"), raw (searchable), contentVector (vector)

Env:
  - AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION (default 2024-12-01-preview)
  - AZURE_TEXT_EMBEDDING_DEPLOYMENT_NAME
  - AZURE_OPENAI_DEPLOYMENT_NAME
  - AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY (or AZURE_SEARCH_KEY)
  - RAPTOR_INDEX (or AZURE_SEARCH_INDEX_NAME) optional, default: raptor-index
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Dict


# ── tiny env loader ────────────────────────────────────────────────
def load_env(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        key = key.strip()
        if key in os.environ:
            continue  # keep existing env (e.g., overriden via CLI)
        os.environ[key] = val


def resolve_imf_text_path() -> Path:
    """Locate the IMF outlook text file: prefer IMF_TEXT_PATH env, else latest IMF_*.txt under data/, else legacy path."""
    env_path = os.getenv("IMF_TEXT_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        sys.exit(f"❌ IMF_TEXT_PATH={env_path} not found.")

    data_dir = Path("data")
    if data_dir.exists():
        candidates = sorted(data_dir.glob("IMF_*.txt"))
        if candidates:
            return candidates[-1]

    legacy = Path("IMF_outlook_oct25.txt")
    if legacy.exists():
        return legacy

    sys.exit("❌ No IMF text found. Place IMF_*.txt under data/ or set IMF_TEXT_PATH.")


# ── HTTP helpers ───────────────────────────────────────────────────
def http_post_json(url: str, headers: dict, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                raise RuntimeError(f"Non-JSON response ({resp.status}): {body[:500]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from None


def http_put_json(url: str, headers: dict, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={**headers, "Content-Type": "application/json"}, method="PUT")
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return json.loads(body) if body else {}
            except json.JSONDecodeError:
                raise RuntimeError(f"Non-JSON response ({resp.status}): {body[:500]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from None


# ── Azure helpers (REST) ───────────────────────────────────────────
def embed_batch(texts: List[str], *, endpoint: str, deployment: str, api_key: str, api_version: str) -> List[List[float]]:
    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={urllib.parse.quote(api_version)}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {"input": texts, "model": deployment}
    resp = http_post_json(url, headers, payload, timeout=60)
    data = resp.get("data") or []
    return [item["embedding"] for item in data]


def chat_summarize(text: str, *, endpoint: str, deployment: str, api_key: str, api_version: str) -> str:
    """Summarize text into a short, high-signal paragraph."""
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={urllib.parse.quote(api_version)}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    def call(payload: dict) -> str:
        resp = http_post_json(url, headers, payload, timeout=60)
        choices = resp.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in summary response: {resp}")
        summary = choices[0].get("message", {}).get("content", "")
        if not summary:
            raise RuntimeError(f"No content in summary response: {resp}")
        return summary.strip()

    base_prompt = (
        "Summarize the following content into a concise paragraph capturing the key points. "
        "Keep it under 120 words.\n\n"
    )

    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": base_prompt + text},
        ],
        "model": deployment,
    }

    try:
        return call(payload)
    except Exception:
        # Retry once with stricter truncation and fewer tokens if the model failed or returned empty.
        payload["messages"][1]["content"] = base_prompt + text[:1200]
        payload.pop("max_completion_tokens", None)
        return call(payload)


# ── vector helpers ──────────────────────────────────────────────────
def l2_normalize(vec: List[float]) -> List[float]:
    s = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / s for x in vec]


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def ensure_index(*, search_endpoint: str, admin_key: str, index_name: str, dims: int = 1536) -> None:
    """Create or update a Search index for RAPTOR nodes."""
    url = f"{search_endpoint}/indexes/{index_name}?api-version=2024-07-01"
    headers = {"api-key": admin_key}
    schema = {
        "name": index_name,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "searchable": False},
            {"name": "level", "type": "Edm.Int32", "searchable": False, "filterable": True, "facetable": True},
            {"name": "kind", "type": "Edm.String", "searchable": False, "filterable": True, "facetable": True},
            {"name": "raw", "type": "Edm.String", "searchable": True, "filterable": False, "facetable": False, "sortable": False},
            {
                "name": "contentVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": dims,
                "vectorSearchProfile": "vprofile",
            },
        ],
        "vectorSearch": {
            "algorithms": [
                {"name": "hnsw-cosine", "kind": "hnsw", "hnswParameters": {"metric": "cosine"}}
            ],
            "profiles": [
                {"name": "vprofile", "algorithm": "hnsw-cosine"}
            ],
        },
    }
    http_put_json(url, headers, schema, timeout=60)


def upload_documents(docs: List[dict], *, search_endpoint: str, admin_key: str, index_name: str) -> None:
    if not docs:
        return
    url = f"{search_endpoint}/indexes/{index_name}/docs/index?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": admin_key}
    payload = {"value": docs}
    http_post_json(url, headers, payload, timeout=60)


# ── text processing ────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    generic = [
        r"\t", r"\r\n", r"\r",
        r"[^\x00-\x7F]+",
        r"<\/?(table|tr|td|ul|li|p|br)>",
        r"\*\*IMPORTANT:\*\*|\*\*NOTE:\*\*",
        r"```|:::|---|--|###|##|#",
    ]
    for pat in generic:
        text = re.sub(pat, " ", text, flags=re.I)

    imf_noise = [
        r"INTERNATIONAL MONETARY FUND",
        r"WORLD\s+ECONOMIC\s+OUTLOOK",
        r"\|\s*April\s+\d{4}|\|\s*October\s+\d{4}",
        r"^CONTENTS$|^DATA$|^PREFACE$|^FOREWORD$|^EXECUTIVE SUMMARY$",
        r"^ASSUMPTIONS AND CONVENTIONS$|^FURTHER INFORMATION$|^ERRATA$",
        r"^Chapter\s+\d+.*$",
        r"^(Table|Figure|Box|Annex)\s+[A-Z0-9].*$",
        r"^\s*[ivxlcdm]+\s*$",
        r"^\s*\d+\s*$",
    ]
    for pat in imf_noise:
        text = re.sub(pat, " ", text, flags=re.I | re.M)

    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def chunk_text(text: str, words_per_chunk: int = 400, overlap_words: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    step = max(words_per_chunk - overlap_words, 1)
    for start in range(0, len(words), step):
        window = words[start : start + words_per_chunk]
        if window:
            chunks.append(" ".join(window))
    return chunks


# ── ingestion pipeline ────────────────────────────────────────────
def main() -> None:
    load_env()

    embed_deploy = os.getenv("AZURE_TEXT_EMBEDDING_DEPLOYMENT_NAME")
    chat_deploy = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    aoai_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    aoai_key = os.getenv("AZURE_OPENAI_API_KEY")
    aoai_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY") or os.getenv("AZURE_SEARCH_KEY")
    index_name = os.getenv("RAPTOR_INDEX", os.getenv("AZURE_SEARCH_INDEX_NAME", "raptor-index"))

    if not all([embed_deploy, chat_deploy, aoai_endpoint, aoai_key, search_endpoint, search_key]):
        sys.exit("❌ Missing Azure env vars. Check .env.")

    raw_path = resolve_imf_text_path()

    print(f"✅ Config loaded; index={index_name}, embed={embed_deploy}, chat={chat_deploy}")

    cleaned = clean_text(raw_path.read_text(encoding="utf-8", errors="ignore"))
    print(f"🔍 Cleaned text length: {len(cleaned):,} chars")

    base_chunks = chunk_text(cleaned, words_per_chunk=400, overlap_words=80)
    if not base_chunks:
        sys.exit("❌ No chunks produced.")
    print(f"📦 Generated {len(base_chunks)} base chunks")

    ensure_index(search_endpoint=search_endpoint, admin_key=search_key, index_name=index_name, dims=1536)
    print("🏗️  Index ensured/updated.")

    # upload base chunks and collect embeddings
    EMB_BATCH = 16
    UPL_BATCH = 100
    docs: List[dict] = []
    base_vecs: List[List[float]] = []
    for i in range(0, len(base_chunks), EMB_BATCH):
        batch = base_chunks[i : i + EMB_BATCH]
        embs = embed_batch(batch, endpoint=aoai_endpoint, deployment=embed_deploy, api_key=aoai_key, api_version=aoai_version)
        base_vecs.extend(embs)
        for j, (txt, emb) in enumerate(zip(batch, embs)):
            docs.append({
                "@search.action": "upload",
                "id": f"raptor_c{(i+j):06d}",
                "level": 0,
                "kind": "chunk",
                "raw": txt,
                "contentVector": emb,
            })
        if len(docs) >= UPL_BATCH:
            upload_documents(docs, search_endpoint=search_endpoint, admin_key=search_key, index_name=index_name)
            print(f"⬆️  Uploaded {len(docs)} base docs (through {(i+len(batch))} chunks)")
            docs = []
    if docs:
        upload_documents(docs, search_endpoint=search_endpoint, admin_key=search_key, index_name=index_name)
        print(f"⬆️  Uploaded final {len(docs)} base docs.")

    # iterative summaries (semantic clustering)
    current_texts = [{"id": f"raptor_c{i:06d}", "raw": txt} for i, txt in enumerate(base_chunks)]
    current_vecs = [l2_normalize(v) for v in base_vecs]
    level = 1
    GROUP_SIZE = 5

    while len(current_texts) > 1:
        summaries: List[dict] = []
        # greedy semantic grouping by cosine similarity
        unused = set(range(len(current_texts)))
        group_count = 0
        while unused:
            # deterministic seed for reproducibility
            seed = min(unused)
            unused.remove(seed)
            # pick nearest neighbors among remaining
            if not unused:
                neighbors = []
            else:
                sims = [(j, dot(current_vecs[seed], current_vecs[j])) for j in unused]
                sims.sort(key=lambda x: x[1], reverse=True)
                take = min(GROUP_SIZE - 1, len(sims))
                neighbors = [sims[k][0] for k in range(take)]
                for j in neighbors:
                    unused.remove(j)
            group_indices = [seed] + neighbors
            joined = "\n\n".join(current_texts[idx]["raw"] for idx in group_indices)
            joined = joined[:1500]
            summary_text = chat_summarize(
                joined,
                endpoint=aoai_endpoint,
                deployment=chat_deploy,
                api_key=aoai_key,
                api_version=aoai_version,
            )
            summaries.append({"id": f"raptor_s{level}_{group_count:05d}", "raw": summary_text})
            group_count += 1

        # embed + upload summaries at this level
        docs = []
        next_vecs: List[List[float]] = []
        for i in range(0, len(summaries), EMB_BATCH):
            batch = summaries[i : i + EMB_BATCH]
            embs = embed_batch([b["raw"] for b in batch], endpoint=aoai_endpoint, deployment=embed_deploy, api_key=aoai_key, api_version=aoai_version)
            for rec, emb in zip(batch, embs):
                docs.append({
                    "@search.action": "upload",
                    "id": rec["id"],
                    "level": level,
                    "kind": "summary",
                    "raw": rec["raw"],
                    "contentVector": emb,
                })
                next_vecs.append(l2_normalize(emb))
            if len(docs) >= UPL_BATCH:
                upload_documents(docs, search_endpoint=search_endpoint, admin_key=search_key, index_name=index_name)
                print(f"⬆️  Uploaded {len(docs)} summaries at level {level}")
                docs = []
        if docs:
            upload_documents(docs, search_endpoint=search_endpoint, admin_key=search_key, index_name=index_name)
            print(f"⬆️  Uploaded final {len(docs)} summaries at level {level}")

        current_texts = summaries
        current_vecs = next_vecs
        level += 1

    print(f"✅ RAPTOR ingestion (semantic clustering) complete. Levels built: {level}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"❌ Failed: {e}")
