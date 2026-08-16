"""Tests for the non-AI automation tools."""

import asyncio
import json

import pytest

from iloptimus.core import automation_tools
from iloptimus.core.tools import BUILTIN_TOOLS, TOOL_ALIASES, execute_tool


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_all_new_tools_are_registered():
    tool_names = {t.name for t in BUILTIN_TOOLS}
    expected = {
        "extract_html_tables", "extract_page_links", "extract_page_metadata",
        "fetch_json_api", "parse_rss_feed", "crawl_site", "batch_fetch",
        "csv_to_json", "json_to_csv", "json_to_jsonl", "jsonl_to_json",
        "chunk_text", "deduplicate_lines", "regex_extract", "regex_replace",
        "compute_hash", "base64_encode", "base64_decode", "text_diff", "text_stats",
        "write_file", "read_file", "list_files", "delete_file", "file_info",
        "create_zip", "extract_zip",
        "syntax_check_python", "syntax_check_javascript",
        "extract_python_functions", "count_lines",
        "batch_regex_extract", "generate_dataset_rows",
        "merge_json_objects", "flatten_json", "sleep", "retry_plan",
    }
    missing = expected - tool_names
    assert not missing, f"Missing tool definitions: {missing}"


def test_tool_aliases_cover_common_names():
    assert TOOL_ALIASES["hash"] == "compute_hash"
    assert TOOL_ALIASES["read"] == "read_file"
    assert TOOL_ALIASES["write"] == "write_file"
    assert TOOL_ALIASES["ls"] == "list_files"
    assert TOOL_ALIASES["zip"] == "create_zip"
    assert TOOL_ALIASES["unzip"] == "extract_zip"
    assert TOOL_ALIASES["diff"] == "text_diff"
    assert TOOL_ALIASES["stats"] == "text_stats"


# ---------------------------------------------------------------------------
# Data transformation tests
# ---------------------------------------------------------------------------


def test_csv_to_json():
    csv_text = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    result = automation_tools.csv_to_json(csv_text)
    assert result["row_count"] == 2
    assert result["headers"] == ["name", "age", "city"]
    assert result["rows"][0]["name"] == "Alice"
    assert result["rows"][1]["city"] == "LA"


def test_json_to_csv():
    json_text = json.dumps([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
    result = automation_tools.json_to_csv(json_text)
    assert result["row_count"] == 2
    assert "name" in result["csv"]
    assert "Alice" in result["csv"]
    assert "Bob" in result["csv"]


def test_json_to_jsonl():
    json_text = json.dumps([{"a": 1}, {"b": 2}, {"c": 3}])
    result = automation_tools.json_to_jsonl(json_text)
    assert result["row_count"] == 3
    lines = result["jsonl"].split("\n")
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"a": 1}


def test_jsonl_to_json():
    jsonl_text = '{"a": 1}\n{"b": 2}\n{"c": 3}'
    result = automation_tools.jsonl_to_json(jsonl_text)
    assert result["row_count"] == 3
    data = json.loads(result["json"])
    assert len(data) == 3
    assert data[0] == {"a": 1}


def test_chunk_text():
    text = "A" * 2500
    result = automation_tools.chunk_text(text, chunk_size=1000, overlap=100)
    assert result["chunk_count"] >= 3
    assert all(len(chunk) <= 1000 for chunk in result["chunks"])


def test_deduplicate_lines():
    text = "hello\nworld\nhello\nfoo\nworld\nbar"
    result = automation_tools.deduplicate_lines(text)
    assert result["duplicates_removed"] == 2
    assert result["unique_lines"] == 4
    assert result["text"].count("hello") == 1


def test_deduplicate_lines_case_insensitive():
    text = "Hello\nhello\nHELLO\nworld"
    result = automation_tools.deduplicate_lines(text, case_sensitive=False)
    assert result["duplicates_removed"] == 2
    assert result["unique_lines"] == 2


def test_regex_extract():
    text = "Contact: alice@example.com and bob@test.org"
    result = automation_tools.regex_extract(text, r"[\w.]+@[\w.]+")
    assert result["match_count"] == 2
    assert "alice@example.com" in result["matches"]
    assert "bob@test.org" in result["matches"]


def test_regex_extract_with_groups():
    text = "2024-01-15 and 2024-03-20"
    result = automation_tools.regex_extract(text, r"(\d{4})-(\d{2})-(\d{2})")
    assert result["match_count"] == 2
    assert result["matches"][0] == ["2024", "01", "15"]


def test_regex_replace():
    text = "Hello world, world is beautiful"
    result = automation_tools.regex_replace(text, r"world", "earth")
    assert result["replacements_made"] == 2
    assert "earth" in result["text"]
    assert "world" not in result["text"]


def test_compute_hash_sha256():
    result = automation_tools.compute_hash("hello", "sha256")
    assert result["algorithm"] == "sha256"
    assert len(result["hash"]) == 64  # sha256 hex digest length
    assert result["hash"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_compute_hash_md5():
    result = automation_tools.compute_hash("hello", "md5")
    assert len(result["hash"]) == 32  # md5 hex digest length


def test_compute_hash_invalid_algorithm():
    with pytest.raises(ValueError, match="Unsupported"):
        automation_tools.compute_hash("hello", "invalid_algo")


def test_base64_encode_decode():
    original = "Hello, World!"
    encoded = automation_tools.base64_encode(original)
    assert encoded["encoded"] == "SGVsbG8sIFdvcmxkIQ=="
    decoded = automation_tools.base64_decode(encoded["encoded"])
    assert decoded["decoded"] == original


def test_text_diff():
    text_a = "line1\nline2\nline3"
    text_b = "line1\nline2_modified\nline3"
    result = automation_tools.text_diff(text_a, text_b)
    assert result["lines_removed"] >= 1
    assert result["lines_added"] >= 1
    assert "line2_modified" in result["diff"]


def test_text_stats():
    text = "Hello world. This is a test. It has multiple sentences!"
    result = automation_tools.text_stats(text)
    assert result["word_count"] == 10
    assert result["sentence_count"] == 3
    assert result["char_count"] == len(text)
    assert result["avg_word_length"] > 0


# ---------------------------------------------------------------------------
# File & workspace operations tests
# ---------------------------------------------------------------------------


def test_write_and_read_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    automation_tools.write_file("test.txt", "Hello, workspace!")
    result = automation_tools.read_file("test.txt")
    assert result["content"] == "Hello, workspace!"
    assert result["size"] == 17


def test_write_file_append(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    automation_tools.write_file("log.txt", "line1\n")
    automation_tools.write_file("log.txt", "line2\n", append=True)
    result = automation_tools.read_file("log.txt")
    assert "line1" in result["content"]
    assert "line2" in result["content"]


def test_list_files(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    automation_tools.write_file("a.txt", "a")
    automation_tools.write_file("b.txt", "b")
    result = automation_tools.list_files(".")
    assert result["entry_count"] >= 2
    names = [e["name"] for e in result["entries"]]
    assert "a.txt" in names
    assert "b.txt" in names


def test_delete_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    automation_tools.write_file("temp.txt", "temp")
    automation_tools.delete_file("temp.txt")
    with pytest.raises(FileNotFoundError):
        automation_tools.read_file("temp.txt")


def test_file_info(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    automation_tools.write_file("data.json", '{"key": "value"}')
    result = automation_tools.file_info("data.json")
    assert result["is_file"] is True
    assert result["extension"] == ".json"
    assert result["size"] > 0


def test_path_traversal_blocked(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    with pytest.raises(ValueError, match="escapes the workspace"):
        automation_tools.read_file("../../etc/passwd")


def test_create_and_extract_zip(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    automation_tools.write_file("file1.txt", "content1")
    automation_tools.write_file("file2.txt", "content2")
    automation_tools.create_zip(["file1.txt", "file2.txt"], "archive.zip")
    # Extract to a subdirectory
    automation_tools.extract_zip("archive.zip", "extracted")
    result = automation_tools.list_files("extracted")
    names = [e["name"] for e in result["entries"]]
    assert "file1.txt" in names
    assert "file2.txt" in names


# ---------------------------------------------------------------------------
# Code analysis tests
# ---------------------------------------------------------------------------


def test_syntax_check_python_valid():
    code = "def hello():\n    return 'world'"
    result = automation_tools.syntax_check_python(code)
    assert result["valid"] is True
    assert result["errors"] == []


def test_syntax_check_python_invalid():
    code = "def hello(:\n    return 'world'"
    result = automation_tools.syntax_check_python(code)
    assert result["valid"] is False
    assert len(result["errors"]) >= 1


def test_syntax_check_javascript_valid():
    code = "function hello() { return 'world'; }"
    result = automation_tools.syntax_check_javascript(code)
    assert result["valid"] is True


def test_syntax_check_javascript_invalid():
    code = "function hello() { return 'world'; "
    result = automation_tools.syntax_check_javascript(code)
    assert result["valid"] is False


def test_syntax_check_javascript_with_strings():
    """Brackets inside strings should not confuse the checker."""
    code = 'var x = "function() { not real }"; var y = 1;'
    result = automation_tools.syntax_check_javascript(code)
    assert result["valid"] is True


def test_extract_python_functions():
    code = '''
def add(a, b):
    """Add two numbers."""
    return a + b

async def fetch(url):
    return url
'''
    result = automation_tools.extract_python_functions(code)
    assert result["function_count"] == 2
    names = [f["name"] for f in result["functions"]]
    assert "add" in names
    assert "fetch" in names
    add_func = next(f for f in result["functions"] if f["name"] == "add")
    assert "a" in add_func["args"]
    assert "Add two numbers" in add_func["docstring"]
    fetch_func = next(f for f in result["functions"] if f["name"] == "fetch")
    assert fetch_func["is_async"] is True


def test_count_lines():
    code = "# comment\n\nx = 1\ny = 2\n# another comment\n"
    result = automation_tools.count_lines(code, "python")
    assert result["total_lines"] == 6
    assert result["comment_lines"] == 2
    assert result["blank_lines"] == 2  # empty line + trailing newline
    assert result["code_lines"] == 2


# ---------------------------------------------------------------------------
# Batch / utility tests
# ---------------------------------------------------------------------------


def test_batch_regex_extract():
    items = ["user@example.com", "no email here", "admin@test.org"]
    result = automation_tools.batch_regex_extract(items, r"[\w.]+@[\w.]+")
    assert result["item_count"] == 3
    assert len(result["results"][0]["matches"]) == 1
    assert len(result["results"][1]["matches"]) == 0
    assert len(result["results"][2]["matches"]) == 1


def test_generate_dataset_rows():
    template = '{"prompt": "What is {topic}?", "answer": "{answer}"}'
    variables = {"topic": ["Python", "Rust"], "answer": ["a language", "a systems language"]}
    result = automation_tools.generate_dataset_rows(template, variables)
    assert result["row_count"] == 4  # 2x2 cartesian product
    lines = result["jsonl"].split("\n")
    assert "Python" in lines[0]
    assert "Rust" in lines[2]


def test_merge_json_objects_shallow():
    objs = ['{"a": 1, "b": 2}', '{"b": 3, "c": 4}']
    result = automation_tools.merge_json_objects(objs, strategy="shallow")
    merged = json.loads(result["merged"])
    assert merged == {"a": 1, "b": 3, "c": 4}  # b overwritten


def test_merge_json_objects_deep():
    objs = ['{"a": {"x": 1, "y": 2}}', '{"a": {"y": 3, "z": 4}}']
    result = automation_tools.merge_json_objects(objs, strategy="deep")
    merged = json.loads(result["merged"])
    assert merged == {"a": {"x": 1, "y": 3, "z": 4}}  # y merged, x and z preserved


def test_flatten_json():
    json_text = json.dumps({"a": {"b": {"c": 1}, "d": 2}, "e": [10, 20]})
    result = automation_tools.flatten_json(json_text)
    flat = json.loads(result["flattened"])
    assert flat["a.b.c"] == 1
    assert flat["a.d"] == 2
    assert flat["e.0"] == 10
    assert flat["e.1"] == 20


def test_sleep_tool():
    result = automation_tools.sleep_tool(0.01)
    assert result["slept_seconds"] == 0.01


def test_sleep_tool_clamps():
    result = automation_tools.sleep_tool(100)
    assert result["slept_seconds"] == 30  # clamped to max


def test_retry_plan():
    result = automation_tools.retry_wrapper("fetch API", max_attempts=3)
    assert result["max_attempts"] == 3
    assert result["strategy"] == "exponential-backoff"
    assert len(result["steps"]) == 3
    assert result["steps"][0]["delay_seconds"] == 1
    assert result["steps"][1]["delay_seconds"] == 2
    assert result["steps"][2]["delay_seconds"] == 4


# ---------------------------------------------------------------------------
# RSS feed parsing tests
# ---------------------------------------------------------------------------


def test_parse_rss_feed():
    rss_xml = '''<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Article 1</title>
      <link>https://example.com/1</link>
      <description>First article</description>
      <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Article 2</title>
      <link>https://example.com/2</link>
      <description>Second article</description>
    </item>
  </channel>
</rss>'''
    result = automation_tools.parse_rss_feed(rss_xml)
    assert result["feed_type"] == "rss"
    assert result["title"] == "Test Feed"
    assert result["entry_count"] == 2
    assert result["entries"][0]["title"] == "Article 1"
    assert result["entries"][1]["link"] == "https://example.com/2"


def test_parse_atom_feed():
    atom_xml = '''<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Test</title>
  <entry>
    <title>Entry 1</title>
    <link href="https://example.com/e1"/>
    <summary>Summary 1</summary>
    <published>2024-01-01T00:00:00Z</published>
  </entry>
</feed>'''
    result = automation_tools.parse_rss_feed(atom_xml)
    assert result["feed_type"] == "atom"
    assert result["title"] == "Atom Test"
    assert result["entry_count"] == 1
    assert result["entries"][0]["link"] == "https://example.com/e1"


# ---------------------------------------------------------------------------
# Execute tool integration tests (verify dispatch works)
# ---------------------------------------------------------------------------


def test_execute_tool_text_stats():
    result = _run(execute_tool("text_stats", {"text": "Hello world."}, {}))
    assert result["ok"] is True
    assert result["result"]["word_count"] == 2


def test_execute_tool_compute_hash():
    result = _run(execute_tool("compute_hash", {"data": "hello", "algorithm": "sha256"}, {}))
    assert result["ok"] is True
    assert len(result["result"]["hash"]) == 64


def test_execute_tool_csv_to_json():
    result = _run(execute_tool("csv_to_json", {"csv_text": "a,b\n1,2"}, {}))
    assert result["ok"] is True
    assert result["result"]["row_count"] == 1


def test_execute_tool_write_read_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    write_result = _run(execute_tool("write_file", {"path": "test.txt", "content": "hello"}, {}))
    assert write_result["ok"] is True
    read_result = _run(execute_tool("read_file", {"path": "test.txt"}, {}))
    assert read_result["ok"] is True
    assert read_result["result"]["content"] == "hello"


def test_execute_tool_syntax_check_python():
    result = _run(execute_tool("syntax_check_python", {"code": "x = 1"}, {}))
    assert result["ok"] is True
    assert result["result"]["valid"] is True


def test_execute_tool_unknown_returns_error():
    result = _run(execute_tool("nonexistent_tool", {}, {}))
    assert result["ok"] is False
    assert "Unknown" in result["error"]


def test_execute_tool_base64_roundtrip():
    encode_result = _run(execute_tool("base64_encode", {"text": "test data"}, {}))
    assert encode_result["ok"] is True
    encoded = encode_result["result"]["encoded"]
    decode_result = _run(execute_tool("base64_decode", {"encoded": encoded}, {}))
    assert decode_result["ok"] is True
    assert decode_result["result"]["decoded"] == "test data"


def test_execute_tool_generate_dataset_rows():
    result = _run(execute_tool(
        "generate_dataset_rows",
        {"template": "{a}+{b}", "variables": {"a": ["1", "2"], "b": ["x", "y"]}},
        {},
    ))
    assert result["ok"] is True
    assert result["result"]["row_count"] == 4


def test_execute_tool_flatten_json():
    result = _run(execute_tool("flatten_json", {"json_text": '{"a": {"b": 1}}'}, {}))
    assert result["ok"] is True
    assert "a.b" in result["result"]["flattened"]


def test_execute_tool_retry_plan():
    result = _run(execute_tool("retry_plan", {"description": "test operation"}, {}))
    assert result["ok"] is True
    assert result["result"]["max_attempts"] == 3
