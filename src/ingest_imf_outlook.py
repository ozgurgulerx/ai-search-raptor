#!/usr/bin/env python3
"""
ingest_imf_outlook.py
─────────────────────
Clean, chunk, embed, and upload IMF_outlook_oct25 content to Azure AI Search.

Dependencies: standard library only (no external pip installs needed).
Relies on Azure OpenAI + Azure AI Search credentials in .env.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Iterable, List, Tuple


# ── tiny .env loader (no python-dotenv) ─────────────────────────────
def load_env(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        os.environ[key.strip()] = val


# ── helpers ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Light cleaning similar to the notebook: remove headers/footers, TOC noise, and hyphen breaks."""
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)  # undo hyphen line breaks
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
    """Split text into overlapping word windows (≈500-token equivalent)."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    step = max(words_per_chunk - overlap_words, 1)
    for start in range(0, len(words), step):
        window = words[start : start + words_per_chunk]
        if not window:
            continue
        chunks.append(" ".join(window))
    return chunks


def http_post_json(url: str, headers: dict, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from None


def http_put_json(url: str, headers: dict, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={**headers, "Content-Type": "application/json"}, method="PUT")
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from None


# ── Azure clients (REST, no SDKs) ───────────────────────────────────
def embed_batch(texts: List[str], *, endpoint: str, deployment: str, api_key: str, api_version: str) -> List[List[float]]:
    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={urllib.parse.quote(api_version)}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {"input": texts, "model": deployment}
    resp = http_post_json(url, headers, payload, timeout=60)
    data = resp.get("data") or []
    return [item["embedding"] for item in data]


def ensure_index(*, search_endpoint: str, admin_key: str, index_name: str, dims: int = 1536) -> None:
    """Create or update the search index with id/raw/contentVector schema."""
    url = f"{search_endpoint}/indexes/{index_name}?api-version=2024-07-01"
    headers = {"api-key": admin_key}
    schema = {
        "name": index_name,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "searchable": False},
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
    http_put_json(url, headers, schema, timeout=30)


def upload_documents(docs: List[dict], *, search_endpoint: str, admin_key: str, index_name: str) -> None:
    url = f"{search_endpoint}/indexes/{index_name}/docs/index?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": admin_key}
    payload = {"value": docs}
    http_post_json(url, headers, payload, timeout=60)


# ── pipeline ────────────────────────────────────────────────────────
def main() -> None:
    load_env()
    # config
    embed_deploy = os.getenv("AZURE_TEXT_EMBEDDING_DEPLOYMENT_NAME")
    aoai_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    aoai_key = os.getenv("AZURE_OPENAI_API_KEY")
    aoai_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY") or os.getenv("AZURE_SEARCH_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "index01")

    if not all([embed_deploy, aoai_endpoint, aoai_key, search_endpoint, search_key]):
        sys.exit("❌ Missing Azure env vars. Check .env for OpenAI and Search settings.")

    print(f"✅ Config loaded; index={index_name}, embed={embed_deploy}")

    # read + clean
    raw_path = Path("IMF_outlook_oct25.txt")
    if not raw_path.exists():
        sys.exit(f"❌ {raw_path} not found. Run pdftotext first.")
    raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw_text)
    print(f"🔍 Cleaned text length: {len(cleaned):,} chars")

    # chunk
    chunks = chunk_text(cleaned, words_per_chunk=400, overlap_words=80)
    if not chunks:
        sys.exit("❌ No chunks produced; check input text.")
    print(f"📦 Generated {len(chunks)} chunks")

    # ensure index
    ensure_index(search_endpoint=search_endpoint, admin_key=search_key, index_name=index_name, dims=1536)
    print("🏗️  Index ensured/updated.")

    # embed + upload in batches
    EMB_BATCH = 16
    UPL_BATCH = 100
    batched_docs: List[dict] = []
    for i in range(0, len(chunks), EMB_BATCH):
        batch = chunks[i : i + EMB_BATCH]
        embs = embed_batch(batch, endpoint=aoai_endpoint, deployment=embed_deploy, api_key=aoai_key, api_version=aoai_version)
        if len(embs) != len(batch):
            sys.exit("❌ Embedding batch size mismatch.")
        for j, (text, emb) in enumerate(zip(batch, embs)):
            doc_id = f"IMF25_c{(i+j):06d}"
            batched_docs.append({
                "@search.action": "upload",
                "id": doc_id,
                "raw": text,
                "contentVector": emb,
            })
        # upload when reaching UPL_BATCH or last
        while len(batched_docs) >= UPL_BATCH:
            send = batched_docs[:UPL_BATCH]
            upload_documents(send, search_endpoint=search_endpoint, admin_key=search_key, index_name=index_name)
            print(f"⬆️  Uploaded {len(send)} docs (through {(i+len(batch))} chunks)")
            batched_docs = batched_docs[UPL_BATCH:]

    if batched_docs:
        upload_documents(batched_docs, search_endpoint=search_endpoint, admin_key=search_key, index_name=index_name)
        print(f"⬆️  Uploaded final {len(batched_docs)} docs.")

    print("✅ Ingestion complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"❌ Failed: {e}")
