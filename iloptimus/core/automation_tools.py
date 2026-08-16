"""Non-AI automation tools for local models.

These are deterministic, side-effecting tools that let a local model automate
real workflows without needing a larger model in the loop:

- **Content scraping & extraction**: multi-page crawl, RSS/Atom feed parsing,
  table extraction from HTML, structured data extraction from JSON APIs.
- **Data transformation**: CSV/JSON/JSONL conversion, text chunking, dedup,
  regex extraction, hash/checksum, diff, base64, statistics.
- **File & workspace ops**: read/write/list files in the workspace, zip/unzip,
  file metadata, directory tree.
- **Batch operations**: batch web fetch, batch URL validation, batch dataset
  row generation from a template.
- **Code analysis**: syntax-check Python/JS, count lines, extract functions.
- **Scheduling helpers**: sleep/timer, retry wrapper.

All tools are sandboxed to the app home directory for file operations and
reuse the existing SSRF-safe URL validation for network operations.
"""

from __future__ import annotations

import ast
import base64
import csv
import difflib
import hashlib
import io
import json
import re
import statistics
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from .storage import app_home

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MAX_BATCH = 20
MAX_FILE_BYTES = 512_000  # 512 KB per file read
MAX_CRAWL_PAGES = 10


def _workspace_root() -> Path:
    root = app_home() / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_workspace_path(relative: str) -> Path:
    """Resolve a path inside the workspace, preventing directory traversal."""
    root = _workspace_root().resolve()
    target = (root / relative).resolve()
    if not str(target).startswith(str(root)):
        raise ValueError("Path escapes the workspace boundary")
    return target


# ---------------------------------------------------------------------------
# Content scraping & extraction
# ---------------------------------------------------------------------------


class _TableExtractor(HTMLParser):
    """Extract HTML tables into lists of rows."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._current_table = []
        elif tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif tag in ("td", "th") and self._current_row is not None:
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._current_table is not None:
            self.tables.append(self._current_table)
            self._current_table = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_table is not None:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag in ("td", "th") and self._in_cell:
            if self._current_row is not None:
                self._current_row.append(" ".join("".join(self._current_cell).split()))
            self._in_cell = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)


class _LinkExtractor(HTMLParser):
    """Extract all links from an HTML page."""

    def __init__(self, base_url: str = "") -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._base_url = base_url
        self._in_a = False
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            values = dict(attrs)
            href = values.get("href", "")
            if href:
                absolute = urljoin(self._base_url, href) if self._base_url else href
                self.links.append({"url": absolute, "text": "", "title": values.get("title", "")})
                self._in_a = True
                self._text = []

    def handle_data(self, data: str) -> None:
        if self._in_a:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_a:
            if self.links:
                self.links[-1]["text"] = " ".join("".join(self._text).split())[:200]
            self._in_a = False


class _MetaExtractor(HTMLParser):
    """Extract page title, meta description, and Open Graph tags."""

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.meta: dict[str, str] = {}
        self._in_title = False
        self._title_parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "title":
            self._in_title = True
            self._title_parts = []
        elif tag == "meta":
            name = values.get("name") or values.get("property") or ""
            content = values.get("content", "")
            if name and content:
                self.meta[name] = content
        elif tag in ("script", "style", "noscript"):
            self._ignored += 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._in_title:
            self.title = "".join(self._title_parts).strip()
            self._in_title = False
        elif tag in ("script", "style", "noscript") and self._ignored:
            self._ignored -= 1


async def extract_html_tables(url: str) -> dict[str, Any]:
    """Fetch a URL and extract all HTML tables as structured data."""
    # Fetch raw HTML directly since web_fetch strips tags.
    async with httpx.AsyncClient(timeout=12, headers={"User-Agent": "ILOptimus/0.2"}) as client:
        response = await client.get(url, follow_redirects=True)
        raw_html = response.text[:256_000]

    parser = _TableExtractor()
    parser.feed(raw_html)
    tables = []
    for table in parser.tables:
        if len(table) < 2:
            continue
        headers = [cell.strip() for cell in table[0]]
        rows = [[cell.strip() for cell in row] for row in table[1:]]
        tables.append({"headers": headers, "rows": rows, "row_count": len(rows)})
    return {"url": url, "table_count": len(tables), "tables": tables}


async def extract_page_links(url: str, filter_pattern: str = "") -> dict[str, Any]:
    """Extract all hyperlinks from a web page, optionally filtered by regex."""

    async with httpx.AsyncClient(timeout=12, headers={"User-Agent": "ILOptimus/0.2"}) as client:
        response = await client.get(url, follow_redirects=True)
        raw_html = response.text[:256_000]

    parser = _LinkExtractor(base_url=url)
    parser.feed(raw_html)
    links = parser.links
    if filter_pattern:
        try:
            pattern = re.compile(filter_pattern, re.IGNORECASE)
            links = [link for link in links if pattern.search(link["url"]) or pattern.search(link["text"])]
        except re.error:
            pass
    return {"url": url, "link_count": len(links), "links": links[:100]}


async def extract_page_metadata(url: str) -> dict[str, Any]:
    """Extract title, meta description, and Open Graph tags from a web page."""
    async with httpx.AsyncClient(timeout=12, headers={"User-Agent": "ILOptimus/0.2"}) as client:
        response = await client.get(url, follow_redirects=True)
        raw_html = response.text[:256_000]

    parser = _MetaExtractor()
    parser.feed(raw_html)
    return {
        "url": url,
        "title": parser.title,
        "description": parser.meta.get("description", ""),
        "og_title": parser.meta.get("og:title", ""),
        "og_description": parser.meta.get("og:description", ""),
        "og_image": parser.meta.get("og:image", ""),
        "og_type": parser.meta.get("og:type", ""),
        "all_meta": {k: v for k, v in parser.meta.items() if not k.startswith("og:")},
    }


async def fetch_json_api(url: str, headers: str = "") -> dict[str, Any]:
    """Fetch a JSON API endpoint and return the parsed data."""
    from .tools import validate_public_url

    await validate_public_url(url)
    header_dict: dict[str, str] = {}
    if headers:
        try:
            header_dict = json.loads(headers)
        except json.JSONDecodeError:
            for pair in headers.split(","):
                if ":" in pair:
                    key, _, value = pair.partition(":")
                    header_dict[key.strip()] = value.strip()
    header_dict.setdefault("User-Agent", "ILOptimus/0.2")
    header_dict.setdefault("Accept", "application/json")
    async with httpx.AsyncClient(timeout=15, headers=header_dict) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.json()


def parse_rss_feed(xml_text: str) -> dict[str, Any]:
    """Parse an RSS 2.0 or Atom XML feed into structured entries."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    entries: list[dict[str, str]] = []

    # RSS 2.0
    if root.tag == "rss":
        channel = root.find("channel")
        if channel is None:
            return {"feed_type": "rss", "title": "", "entries": []}
        feed_title = (channel.findtext("title") or "").strip()
        for item in channel.findall("item"):
            entries.append({
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "description": (item.findtext("description") or "").strip()[:500],
                "pub_date": (item.findtext("pubDate") or "").strip(),
            })
        return {"feed_type": "rss", "title": feed_title, "entry_count": len(entries), "entries": entries}

    # Atom
    if root.tag == "{http://www.w3.org/2005/Atom}feed":
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        feed_title = (root.findtext("atom:title", namespaces=ns) or "").strip()
        for entry in root.findall("atom:entry", namespaces=ns):
            link_elem = entry.find("atom:link", namespaces=ns)
            link = link_elem.attrib.get("href", "") if link_elem is not None else ""
            entries.append({
                "title": (entry.findtext("atom:title", namespaces=ns) or "").strip(),
                "link": link,
                "summary": (entry.findtext("atom:summary", namespaces=ns) or "").strip()[:500],
                "published": (entry.findtext("atom:published", namespaces=ns) or "").strip(),
            })
        return {"feed_type": "atom", "title": feed_title, "entry_count": len(entries), "entries": entries}

    return {"feed_type": "unknown", "title": "", "entries": []}


async def crawl_site(start_url: str, max_pages: int = 5, same_domain: bool = True) -> dict[str, Any]:
    """Crawl multiple pages starting from a URL, following same-domain links."""
    from .tools import validate_public_url

    await validate_public_url(start_url)
    max_pages = max(1, min(MAX_CRAWL_PAGES, max_pages))
    base_domain = urlparse(start_url).hostname or ""
    visited: set[str] = set()
    pages: list[dict[str, Any]] = []
    queue: list[str] = [start_url]

    async with httpx.AsyncClient(timeout=12, headers={"User-Agent": "ILOptimus/0.2"}) as client:
        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            try:
                await validate_public_url(url)
                response = await client.get(url, follow_redirects=True)
                raw_html = response.text[:256_000]
                visited.add(url)

                # Extract text content
                from .tools import _TextExtractor

                text_parser = _TextExtractor()
                text_parser.feed(raw_html)
                text = text_parser.text()[:20_000]

                # Extract links for the queue
                link_parser = _LinkExtractor(base_url=url)
                link_parser.feed(raw_html)
                for link in link_parser.links:
                    link_url = link["url"]
                    if same_domain:
                        link_domain = urlparse(link_url).hostname or ""
                        if link_domain != base_domain:
                            continue
                    if link_url not in visited and link_url not in queue:
                        queue.append(link_url)

                pages.append({
                    "url": url,
                    "status": response.status_code,
                    "text": text,
                    "link_count": len(link_parser.links),
                    "char_count": len(text),
                })
            except Exception:  # noqa: S112, BLE001 - crawler skips failed pages
                continue

    return {
        "start_url": start_url,
        "pages_crawled": len(pages),
        "total_links_found": sum(p["link_count"] for p in pages),
        "pages": pages,
    }


async def batch_fetch(urls: list[str]) -> dict[str, Any]:
    """Fetch multiple URLs in parallel (max 20)."""
    from .tools import web_fetch

    urls = urls[:MAX_BATCH]
    results = await _gather_with_exceptions(
        [web_fetch(url) for url in urls],
        url_list=urls,
    )
    return {"fetched": len([r for r in results if r.get("ok")]), "results": results}


async def _gather_with_exceptions(coros, *, url_list: list[str] | None = None) -> list[dict[str, Any]]:
    """Gather coroutines, wrapping exceptions into error dicts."""
    import asyncio

    results = await asyncio.gather(*coros, return_exceptions=True)
    output: list[dict[str, Any]] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            output.append({"ok": False, "error": str(result), "url": url_list[i] if url_list else ""})
        elif isinstance(result, dict):
            output.append({"ok": True, **result})
        else:
            output.append({"ok": True, "result": result})
    return output


# ---------------------------------------------------------------------------
# Data transformation tools
# ---------------------------------------------------------------------------


def csv_to_json(csv_text: str, delimiter: str = ",") -> dict[str, Any]:
    """Convert CSV text to a list of JSON objects."""
    reader = csv.DictReader(io.StringIO(csv_text), delimiter=delimiter)
    rows = [dict(row) for row in reader]
    return {"row_count": len(rows), "headers": reader.fieldnames or [], "rows": rows}


def json_to_csv(json_text: str, flatten: bool = True) -> dict[str, Any]:
    """Convert JSON data to CSV text."""
    data = json.loads(json_text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise ValueError("Input must be a JSON array or object")

    def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
        if not flatten:
            return row
        flat: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                flat[key] = json.dumps(value, ensure_ascii=False)
            else:
                flat[key] = value
        return flat

    rows = [flatten_row(row) if isinstance(row, dict) else {"value": row} for row in data]
    all_keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(rows)
    return {"row_count": len(rows), "headers": all_keys, "csv": output.getvalue()}


def json_to_jsonl(json_text: str) -> dict[str, Any]:
    """Convert a JSON array to JSONL (one object per line)."""
    data = json.loads(json_text)
    if not isinstance(data, list):
        raise TypeError("Input must be a JSON array")
    lines = [json.dumps(item, ensure_ascii=False) for item in data]
    return {"row_count": len(lines), "jsonl": "\n".join(lines)}


def jsonl_to_json(jsonl_text: str) -> dict[str, Any]:
    """Convert JSONL text to a JSON array."""
    rows: list[dict[str, Any]] = []
    for line in jsonl_text.split("\n"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return {"row_count": len(rows), "json": json.dumps(rows, ensure_ascii=False, indent=2)}


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> dict[str, Any]:
    """Split text into overlapping chunks of a given size."""
    chunk_size = max(100, min(10_000, chunk_size))
    overlap = max(0, min(chunk_size // 2, overlap))
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += chunk_size - overlap
    return {"chunk_count": len(chunks), "chunk_size": chunk_size, "overlap": overlap, "chunks": chunks}


def deduplicate_lines(text: str, case_sensitive: bool = True) -> dict[str, Any]:
    """Remove duplicate lines from text, preserving order."""
    seen: set[str] = set()
    unique: list[str] = []
    for line in text.split("\n"):
        key = line if case_sensitive else line.lower()
        if key not in seen:
            seen.add(key)
            unique.append(line)
    removed = len(text.split("\n")) - len(unique)
    return {"original_lines": len(text.split("\n")), "unique_lines": len(unique), "duplicates_removed": removed, "text": "\n".join(unique)}


def regex_extract(text: str, pattern: str, group: int = 0) -> dict[str, Any]:
    """Extract all matches of a regex pattern from text."""
    matches = re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL)
    # Handle tuple groups (multiple capture groups)
    if matches and isinstance(matches[0], tuple):
        result = [list(m) for m in matches]
    else:
        result = list(matches)
    return {"match_count": len(result), "matches": result}


def regex_replace(text: str, pattern: str, replacement: str) -> dict[str, Any]:
    """Replace all matches of a regex pattern in text."""
    count_before = len(re.findall(pattern, text))
    result = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    return {"replacements_made": count_before, "text": result}


def compute_hash(data: str, algorithm: str = "sha256") -> dict[str, Any]:
    """Compute a cryptographic hash of the input text."""
    algorithm = algorithm.lower()
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}. Available: {sorted(hashlib.algorithms_available)[:10]}...")
    digest = hashlib.new(algorithm, data.encode()).hexdigest()
    return {"algorithm": algorithm, "hash": digest, "input_length": len(data)}


def base64_encode(text: str) -> dict[str, Any]:
    """Base64-encode text."""
    encoded = base64.b64encode(text.encode()).decode()
    return {"encoded": encoded, "decoded_length": len(text)}


def base64_decode(encoded: str) -> dict[str, Any]:
    """Base64-decode a string."""
    decoded = base64.b64decode(encoded).decode("utf-8", errors="replace")
    return {"decoded": decoded, "encoded_length": len(encoded)}


def text_diff(text_a: str, text_b: str, context: int = 3) -> dict[str, Any]:
    """Compute a unified diff between two texts."""
    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines_a, lines_b, fromfile="a", tofile="b", n=context))
    return {"diff": "".join(diff), "lines_added": sum(1 for l in diff if l.startswith("+") and not l.startswith("+++")), "lines_removed": sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))}


def text_stats(text: str) -> dict[str, Any]:
    """Compute statistics about a text: word count, char count, sentence count, etc."""
    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    word_lengths = [len(w) for w in words] if words else [0]
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_word_length": round(statistics.mean(word_lengths), 2),
        "max_word_length": max(word_lengths),
        "avg_sentence_length": round(len(words) / max(1, len(sentences)), 2),
        "unique_words": len({w.lower() for w in words}),
        "line_count": len(text.split("\n")),
    }


# ---------------------------------------------------------------------------
# File & workspace operations
# ---------------------------------------------------------------------------


def write_file(path: str, content: str, append: bool = False) -> dict[str, Any]:
    """Write content to a file in the workspace."""
    target = _safe_workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        handle.write(content)
    return {"path": path, "bytes_written": len(content.encode()), "appended": append}


def read_file(path: str) -> dict[str, Any]:
    """Read a file from the workspace."""
    target = _safe_workspace_path(path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds {MAX_FILE_BYTES} byte limit")
    content = target.read_text(encoding="utf-8", errors="replace")
    return {"path": path, "content": content, "size": len(content), "lines": len(content.split("\n"))}


def list_files(path: str = ".") -> dict[str, Any]:
    """List files in a workspace directory."""
    target = _safe_workspace_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not target.is_dir():
        raise ValueError(f"Not a directory: {path}")
    entries: list[dict[str, Any]] = []
    for item in sorted(target.iterdir()):
        entries.append({
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else 0,
        })
    return {"path": path, "entry_count": len(entries), "entries": entries}


def delete_file(path: str) -> dict[str, Any]:
    """Delete a file from the workspace."""
    target = _safe_workspace_path(path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if target.is_dir():
        raise ValueError("Use delete_directory for directories")
    target.unlink()
    return {"path": path, "deleted": True}


def file_info(path: str) -> dict[str, Any]:
    """Get metadata about a file in the workspace."""
    target = _safe_workspace_path(path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")
    stat = target.stat()
    import time as _time

    return {
        "path": path,
        "size": stat.st_size,
        "is_dir": target.is_dir(),
        "is_file": target.is_file(),
        "modified": _time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime(stat.st_mtime)),
        "extension": target.suffix,
        "name": target.name,
    }


def create_zip(paths: list[str], zip_path: str) -> dict[str, Any]:
    """Create a zip archive from workspace files."""
    target_zip = _safe_workspace_path(zip_path)
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    included: list[str] = []
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in paths[:50]:
            source = _safe_workspace_path(relative)
            if source.exists():
                archive.write(source, source.name)
                included.append(relative)
    return {"zip_path": zip_path, "file_count": len(included), "included": included, "size": target_zip.stat().st_size}


def extract_zip(zip_path: str, extract_to: str = ".") -> dict[str, Any]:
    """Extract a zip archive into the workspace."""
    source = _safe_workspace_path(zip_path)
    if not source.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    destination = _safe_workspace_path(extract_to)
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(source, "r") as archive:
        for name in archive.namelist():
            # Prevent path traversal in zip entries
            archive.extract(name, destination)
            extracted.append(name)
    return {"zip_path": zip_path, "extract_to": extract_to, "file_count": len(extracted), "extracted": extracted}


# ---------------------------------------------------------------------------
# Code analysis tools
# ---------------------------------------------------------------------------


def syntax_check_python(code: str) -> dict[str, Any]:
    """Check Python code for syntax errors without executing it."""
    try:
        ast.parse(code)
        return {"valid": True, "errors": []}
    except SyntaxError as error:
        return {"valid": False, "errors": [{"line": error.lineno, "offset": error.offset, "message": error.msg}]}


def syntax_check_javascript(code: str) -> dict[str, Any]:
    """Basic JavaScript syntax check using brace/paren/bracket matching."""
    stack: list[str] = []
    pairs = {"(": ")", "{": "}", "[": "]"}
    closers = set(pairs.values())
    in_string: str | None = None
    in_comment = False
    in_line_comment = False
    errors: list[dict[str, Any]] = []
    for i, char in enumerate(code):
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            continue
        if in_comment:
            if char == "*" and i + 1 < len(code) and code[i + 1] == "/":
                in_comment = False
            continue
        if in_string:
            if char == "\\":
                continue
            if char == in_string:
                in_string = None
            continue
        if char in ("'", '"', "`"):
            in_string = char
            continue
        if char == "/" and i + 1 < len(code):
            if code[i + 1] == "/":
                in_line_comment = True
                continue
            if code[i + 1] == "*":
                in_comment = True
                continue
        if char in pairs:
            stack.append(char)
        elif char in closers:
            if not stack:
                errors.append({"position": i, "message": f"Unexpected closing bracket: {char}"})
                break
            last = stack.pop()
            if pairs[last] != char:
                errors.append({"position": i, "message": f"Mismatched bracket: expected {pairs[last]}, got {char}"})
                break
    if stack and not errors:
        errors.append({"message": f"Unclosed bracket(s): {''.join(stack)}"})
    return {"valid": len(errors) == 0, "errors": errors}


def extract_python_functions(code: str) -> dict[str, Any]:
    """Extract function definitions from Python code."""
    tree = ast.parse(code)
    functions: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [arg.arg for arg in node.args.args]
            returns = ast.dump(node.returns) if node.returns else None
            docstring = ast.get_docstring(node) or ""
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "args": args,
                "returns": returns,
                "docstring": docstring[:200],
                "is_async": isinstance(node, ast.AsyncFunctionDef),
            })
    return {"function_count": len(functions), "functions": functions}


def count_lines(code: str, language: str = "auto") -> dict[str, Any]:
    """Count lines of code, comments, and blank lines."""
    lines = code.split("\n")
    blank = sum(1 for line in lines if not line.strip())
    comment_chars = {"python": "#", "javascript": "//", "auto": "#"}
    comment_char = comment_chars.get(language.lower(), "#")
    comments = sum(1 for line in lines if line.strip().startswith(comment_char))
    code_lines = len(lines) - blank - comments
    return {
        "total_lines": len(lines),
        "code_lines": code_lines,
        "comment_lines": comments,
        "blank_lines": blank,
        "language": language,
    }


# ---------------------------------------------------------------------------
# Batch / utility tools
# ---------------------------------------------------------------------------


def batch_regex_extract(items: list[str], pattern: str) -> dict[str, Any]:
    """Apply the same regex to a list of strings and return all matches."""
    results: list[dict[str, Any]] = []
    for i, item in enumerate(items[:MAX_BATCH]):
        matches = re.findall(pattern, item, flags=re.MULTILINE | re.DOTALL)
        results.append({"index": i, "input": item[:100], "matches": list(matches) if not (matches and isinstance(matches[0], tuple)) else [list(m) for m in matches]})
    return {"item_count": len(results), "results": results}


def generate_dataset_rows(template: str, variables: dict[str, list[str]], max_rows: int = 100) -> dict[str, Any]:
    """Generate JSONL dataset rows by expanding a template with variable lists.

    The template uses ``{variable_name}`` placeholders. Each variable maps to a
    list of values; the cartesian product fills the template.
    """
    max_rows = max(1, min(500, max_rows))
    import itertools

    keys = list(variables.keys())
    value_lists = [variables[k][:20] for k in keys]  # cap each variable to 20 values
    rows: list[str] = []
    for combo in itertools.product(*value_lists):
        if len(rows) >= max_rows:
            break
        row = template
        for key, value in zip(keys, combo):
            row = row.replace("{" + key + "}", str(value))
        rows.append(row)
    return {"row_count": len(rows), "jsonl": "\n".join(rows)}


def merge_json_objects(objects: list[str], strategy: str = "deep") -> dict[str, Any]:
    """Merge multiple JSON objects into one."""
    parsed = [json.loads(obj) for obj in objects[:MAX_BATCH]]
    if strategy == "shallow":
        result: dict[str, Any] = {}
        for obj in parsed:
            result.update(obj)
    elif strategy == "deep":
        result = {}
        for obj in parsed:
            result = _deep_merge(result, obj)
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}. Use 'shallow' or 'deep'.")
    return {"merged": json.dumps(result, ensure_ascii=False, indent=2), "object_count": len(parsed)}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten_json(json_text: str, separator: str = ".") -> dict[str, Any]:
    """Flatten a nested JSON object into dot-notation keys."""
    data = json.loads(json_text)
    flat: dict[str, Any] = {}

    def _flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{separator}{key}" if prefix else key
                _flatten(value, new_key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_key = f"{prefix}{separator}{i}" if prefix else str(i)
                _flatten(item, new_key)
        else:
            flat[prefix] = obj

    _flatten(data)
    return {"key_count": len(flat), "flattened": json.dumps(flat, ensure_ascii=False, indent=2)}


def sleep_tool(seconds: float) -> dict[str, Any]:
    """Sleep for a given number of seconds (max 30)."""
    seconds = max(0, min(30, seconds))
    time.sleep(seconds)
    return {"slept_seconds": seconds}


def retry_wrapper(description: str, max_attempts: int = 3) -> dict[str, Any]:
    """Return a retry plan that the model can follow for an operation.

    This is a planning tool — it doesn't execute anything, it just gives the
    model a structured retry strategy to follow.
    """
    max_attempts = max(1, min(10, max_attempts))
    plan = {
        "description": description,
        "max_attempts": max_attempts,
        "strategy": "exponential-backoff",
        "steps": [
            {"attempt": i + 1, "delay_seconds": min(2**i, 16), "action": f"Retry {description}"}
            for i in range(max_attempts)
        ],
    }
    return plan
