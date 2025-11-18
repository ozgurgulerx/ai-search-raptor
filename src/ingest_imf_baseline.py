#!/usr/bin/env -S .venv/bin/python
"""
ingest_imf_baseline.py
Build a flat hybrid/vector index (imf_baseline) in Azure AI Search from all IMF_*.pdf files.
Workflow: pdftotext -> clean -> chunk (~400 words, 80 overlap) -> embed -> upload with doc_id metadata.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List


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


def embed_batch(texts: List[str], *, endpoint: str, deployment: str, api_key: str, api_version: str) -> List[List[float]]:
    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={urllib.parse.quote(api_version)}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {"input": texts, "model": deployment}
    for attempt in range(5):
        try:
            resp = http_post_json(url, headers, payload, timeout=60)
            break
        except RuntimeError as e:
            msg = str(e)
            if "429" in msg or "RateLimit" in msg:
                wait = 5 * (attempt + 1)
                print(f"   ⏳ Rate limit, retrying in {wait}s …")
                time.sleep(wait)
                continue
            raise
    else:
        raise RuntimeError("Exceeded retry attempts for embeddings.")
    data = resp.get("data") or []
    return [item["embedding"] for item in data]


def ensure_index(*, search_endpoint: str, admin_key: str, index_name: str, dims: int = 1536) -> None:
    url = f"{search_endpoint}/indexes/{index_name}?api-version=2024-07-01"
    headers = {"api-key": admin_key}
    schema = {
        "name": index_name,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "searchable": False},
            {"name": "doc_id", "type": "Edm.String", "searchable": False, "filterable": True, "facetable": True},
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
    for attempt in range(5):
        try:
            http_post_json(url, headers, payload, timeout=60)
            return
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RateLimit" in msg or "Remote end closed" in msg:
                wait = 5 * (attempt + 1)
                print(f"   ⏳ Upload retry in {wait}s … ({msg[:80]})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Exceeded retry attempts for upload.")


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


def ensure_txt_from_pdf(pdf_path: Path) -> Path:
    txt_path = pdf_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path
    if not shutil.which("pdftotext"):
        sys.exit("❌ pdftotext is required but not found.")
    subprocess.run(["pdftotext", str(pdf_path), str(txt_path)], check=True)
    return txt_path


def main() -> None:
    load_env()

    embed_deploy = os.getenv("AZURE_TEXT_EMBEDDING_DEPLOYMENT_NAME")
    aoai_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    aoai_key = os.getenv("AZURE_OPENAI_API_KEY")
    aoai_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY") or os.getenv("AZURE_SEARCH_KEY")
    index_name = "imf_baseline"

    if not all([embed_deploy, aoai_endpoint, aoai_key, search_endpoint, search_key]):
        sys.exit("❌ Missing Azure env vars. Check .env.")

    pdf_files = sorted(Path(".").glob("IMF_*.pdf"))
    if not pdf_files:
        sys.exit("❌ No IMF_*.pdf files found.")

    ensure_index(search_endpoint=search_endpoint, admin_key=search_key, index_name=index_name, dims=1536)
    print(f"🏗️  Index ensured: {index_name}")

    EMB_BATCH = 16
    UPL_BATCH = 100

    for pdf in pdf_files:
        doc_id = pdf.stem
        print(f"📄 Processing {pdf.name} ({doc_id}) …")
        try:
            txt_path = ensure_txt_from_pdf(pdf)
        except Exception as e:
            print(f"⚠️  Skipping {pdf.name}: {e}")
            continue
        raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned, words_per_chunk=400, overlap_words=80)
        print(f"   → {len(chunks)} chunks")
        if not chunks:
            continue

        docs: List[dict] = []
        for i in range(0, len(chunks), EMB_BATCH):
            batch = chunks[i : i + EMB_BATCH]
            embs = embed_batch(batch, endpoint=aoai_endpoint, deployment=embed_deploy, api_key=aoai_key, api_version=aoai_version)
            for j, (txt, emb) in enumerate(zip(batch, embs)):
                docs.append({
                    "@search.action": "upload",
                    "id": f"{doc_id}_c{(i+j):05d}",
                    "doc_id": doc_id,
                    "raw": txt,
                    "contentVector": emb,
                })
            if len(docs) >= UPL_BATCH:
                upload_documents(docs, search_endpoint=search_endpoint, admin_key=search_key, index_name=index_name)
                print(f"   ⬆️  Uploaded {len(docs)} docs so far")
                docs = []
        if docs:
            upload_documents(docs, search_endpoint=search_endpoint, admin_key=search_key, index_name=index_name)
            print(f"   ⬆️  Uploaded final {len(docs)} docs for {doc_id}")

    print("✅ Baseline ingestion complete.")


if __name__ == "__main__":
    try:
        import shutil
        main()
    except Exception as e:
        sys.exit(f"❌ Failed: {e}")
