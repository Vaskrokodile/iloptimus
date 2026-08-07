"""Handcrafted coding tasks for the IL (Intuition Learning) pipeline.

Each task is a single, handcrafted scenario (NOT procedurally generated) with:
- a spec the model reads
- a function signature it must implement
- hidden test cases the grader runs in a sandbox
- expected reasoning concepts (for reasoning-quality scoring)
- an anti-laziness profile (degenerate-solution detectors)

The model responds in <reasoning>...</reasoning><answer>```python\n...\n```</answer>
format. The reward runs the code in a sandbox, scores test pass-rate with an
anti-laziness penalty, then shapes the final reward by reasoning quality:

    final = correctness * (0.6 + 0.4 * reasoning_quality)

This is the IL efficiency-aware reward shaping — wrong answers always get 0,
right answers with lazy reasoning get 0.6, right answers with thorough verified
reasoning get up to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodingTask:
    """A single handcrafted coding task."""
    idx: int
    name: str
    spec: str
    signature: str
    tests: list[str]
    expected_concepts: list[str]
    token_budget: int = 600
    difficulty: str = "medium"
    # anti-laziness: names of params the body must reference
    required_params: list[str] = field(default_factory=list)
    # anti-laziness: the solution must contain at least one of these
    required_constructs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 12 handcrafted coding tasks
# ---------------------------------------------------------------------------

TASKS: list[CodingTask] = [
    CodingTask(
        idx=0,
        name="merge_sorted_lists",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Implement `merge_sorted(a, b)` which merges two already-sorted "
            "lists of integers into one sorted list. Do NOT call `sorted()` on "
            "the concatenation — walk both lists with two pointers. Duplicates "
            "from both lists must be preserved."
        ),
        signature="def merge_sorted(a: list[int], b: list[int]) -> list[int]:",
        tests=[
            "assert merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]",
            "assert merge_sorted([], []) == []",
            "assert merge_sorted([1], []) == [1]",
            "assert merge_sorted([], [2]) == [2]",
            "assert merge_sorted([1,1,2], [1,2,3]) == [1,1,1,2,2,3]",
            "assert merge_sorted([5,6,7], [1,2,3]) == [1,2,3,5,6,7]",
            "assert merge_sorted([1,2,3], [4,5,6]) == [1,2,3,4,5,6]",
            "assert merge_sorted([-3,-1], [-2,0]) == [-3,-2,-1,0]",
            "assert merge_sorted([1], [1]) == [1,1]",
            "assert merge_sorted([1,2,2,3], [2,3,3,4]) == [1,2,2,2,3,3,3,4]",
        ],
        expected_concepts=["pointer", "two", "compare", "append", "merge", "sorted"],
        required_params=["a", "b"],
        required_constructs=["while", "append"],
    ),
    CodingTask(
        idx=1,
        name="lru_cache",
        difficulty="hard",
        token_budget=800,
        spec=(
            "Implement an LRU (Least Recently Used) cache class `LRUCache` with "
            "`__init__(capacity)`, `get(key)` returning the value or -1, and "
            "`put(key, value)`. When capacity is exceeded, evict the least "
            "recently used key. Both get and put must be O(1) on average — use "
            "an OrderedDict or a dict + doubly-linked structure, NOT a list scan."
        ),
        signature="class LRUCache:",
        tests=[
            "c = LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1",
            "c.put(3,3); assert c.get(2)==-1",
            "c.put(4,4); assert c.get(1)==-1",
            "assert c.get(3)==3; assert c.get(4)==4",
            "c2 = LRUCache(1); c2.put(1,1); c2.put(2,2); assert c2.get(1)==-1",
            "c3 = LRUCache(2); c3.put(1,1); c3.put(2,2); c3.get(1); c3.put(3,3); assert c3.get(2)==-1",
            "c4 = LRUCache(3); [c4.put(i,i) for i in range(3)]; c4.get(0); c4.put(3,3); assert c4.get(1)==1",
            "c5 = LRUCache(2); c5.put(1,1); c5.put(2,2); c5.put(1,10); assert c5.get(1)==10",
            "c6 = LRUCache(2); c6.put(1,1); c6.put(2,2); c6.get(1); c6.put(2,20); assert c6.get(2)==20",
            "c7 = LRUCache(1); assert c7.get(0)==-1",
        ],
        expected_concepts=["order", "capacity", "evict", "recently", "o(1)", "linked", "dict"],
        required_params=["capacity"],
        required_constructs=["def", "class"],
    ),
    CodingTask(
        idx=2,
        name="off_by_one_fix",
        difficulty="medium",
        token_budget=500,
        spec=(
            "The function `count_vowels(s)` below has an off-by-one bug: it skips "
            "the last character of the string. Fix it so it counts every vowel "
            "(a, e, i, o, u — case-insensitive) in the ENTIRE string.\n\n"
            "Buggy code:\n"
            "```python\n"
            "def count_vowels(s):\n"
            "    count = 0\n"
            "    for i in range(len(s) - 1):\n"
            "        if s[i] in 'aeiouAEIOU':\n"
            "            count += 1\n"
            "    return count\n"
            "```\n"
            "Return the FIXED function."
        ),
        signature="def count_vowels(s: str) -> int:",
        tests=[
            "assert count_vowels('hello') == 2",
            "assert count_vowels('a') == 1",
            "assert count_vowels('') == 0",
            "assert count_vowels('bcdfg') == 0",
            "assert count_vowels('AEIOU') == 5",
            "assert count_vowels('racecar') == 3",
            "assert count_vowels('queue') == 4",
            "assert count_vowels('aAeEiIoOuU') == 10",
            "assert count_vowels('xyz') == 0",
            "assert count_vowels('abababa') == 4",
        ],
        expected_concepts=["off-by-one", "range", "last", "len", "vowel", "fix", "boundary"],
        required_params=["s"],
        required_constructs=["for", "in"],
    ),
    CodingTask(
        idx=3,
        name="binary_search",
        difficulty="medium",
        token_budget=500,
        spec=(
            "Implement `binary_search(arr, target)` returning the index of "
            "`target` in a sorted list `arr`, or -1 if not found. Must use "
            "binary search (O(log n)), NOT `arr.index()`. Handle empty lists "
            "and targets not present."
        ),
        signature="def binary_search(arr: list[int], target: int) -> int:",
        tests=[
            "assert binary_search([1,2,3,4,5], 3) == 2",
            "assert binary_search([1,2,3,4,5], 1) == 0",
            "assert binary_search([1,2,3,4,5], 5) == 4",
            "assert binary_search([1,2,3,4,5], 6) == -1",
            "assert binary_search([], 1) == -1",
            "assert binary_search([1], 1) == 0",
            "assert binary_search([1], 2) == -1",
            "assert binary_search([1,3,5,7,9,11], 7) == 3",
            "assert binary_search([1,3,5,7,9,11], 0) == -1",
            "assert binary_search([1,3,5,7,9,11], 12) == -1",
            "assert binary_search([-5,-3,-1,0,2,4], -3) == 1",
            "assert binary_search([2,2,2,2,2], 2) in range(5)",
        ],
        expected_concepts=["binary", "mid", "left", "right", "log", "search", "sorted"],
        required_params=["arr", "target"],
        required_constructs=["while", "//"],
    ),
    CodingTask(
        idx=4,
        name="validate_parens",
        difficulty="medium",
        token_budget=500,
        spec=(
            "Implement `is_valid_parens(s)` returning True if the string `s` "
            "contains properly matched and nested parentheses `()`, brackets "
            "`[]`, and braces `{}`. Return False otherwise. Use a stack — do "
            "NOT use regex. An empty string is valid."
        ),
        signature="def is_valid_parens(s: str) -> bool:",
        tests=[
            "assert is_valid_parens('()') == True",
            "assert is_valid_parens('()[]{}') == True",
            "assert is_valid_parens('(]') == False",
            "assert is_valid_parens('([)]') == False",
            "assert is_valid_parens('{[]}') == True",
            "assert is_valid_parens('') == True",
            "assert is_valid_parens('(') == False",
            "assert is_valid_parens(')') == False",
            "assert is_valid_parens('((()))') == True",
            "assert is_valid_parens('((())') == False",
            "assert is_valid_parens('()()()') == True",
            "assert is_valid_parens('([{}])') == True",
        ],
        expected_concepts=["stack", "match", "push", "pop", "pair", "nest", "bracket"],
        required_params=["s"],
        required_constructs=["for", "if"],
    ),
    CodingTask(
        idx=5,
        name="word_frequency",
        difficulty="medium",
        token_budget=600,
        spec=(
            "Implement `word_frequency(text, n)` that takes a string of "
            "space-separated words and returns the `n` most common words as a "
            "list of (word, count) tuples, sorted by count descending then "
            "alphabetically for ties. Case-insensitive. Punctuation attached "
            "to words should be stripped. Use collections.Counter."
        ),
        signature="def word_frequency(text: str, n: int) -> list[tuple[str, int]]:",
        tests=[
            "assert word_frequency('the cat the dog the bird', 2) == [('the', 3), ('bird', 1)]",
            "assert word_frequency('a a a b b c', 2) == [('a', 3), ('b', 2)]",
            "assert word_frequency('hello world', 1) == [('hello', 1)]",
            "assert word_frequency('', 5) == []",
            "assert word_frequency('The the THE', 1) == [('the', 3)]",
            "assert word_frequency('cat dog bird', 3) == [('bird', 1), ('cat', 1), ('dog', 1)]",
            "assert word_frequency('a b a b a b', 1) == [('a', 3)]",
            "assert word_frequency('x y z x y z x', 2) == [('x', 3), ('y', 2)]",
        ],
        expected_concepts=["counter", "frequency", "sort", "count", "case", "lower", "strip"],
        required_params=["text", "n"],
        required_constructs=["Counter", "for"],
    ),
    CodingTask(
        idx=6,
        name="matrix_rotate",
        difficulty="hard",
        token_budget=700,
        spec=(
            "Implement `rotate_matrix_90(matrix)` that rotates an N x N matrix "
            "(list of lists) 90 degrees clockwise IN PLACE and returns it. Do "
            "NOT create a new matrix — transpose then reverse each row. Handle "
            "1x1 and empty matrices."
        ),
        signature="def rotate_matrix_90(matrix: list[list[int]]) -> list[list[int]]:",
        tests=[
            "assert rotate_matrix_90([[1,2],[3,4]]) == [[3,1],[4,2]]",
            "assert rotate_matrix_90([[1,2,3],[4,5,6],[7,8,9]]) == [[7,4,1],[8,5,2],[9,6,3]]",
            "assert rotate_matrix_90([[1]]) == [[1]]",
            "assert rotate_matrix_90([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]) == [[13,9,5,1],[14,10,6,2],[15,11,7,3],[16,12,8,4]]",
            "assert rotate_matrix_90([[0,0],[0,0]]) == [[0,0],[0,0]]",
            "m = [[1,2],[3,4]]; assert rotate_matrix_90(rotate_matrix_90(m)) == [[4,3],[2,1]]",
        ],
        expected_concepts=["transpose", "reverse", "in place", "clockwise", "row", "90"],
        required_params=["matrix"],
        required_constructs=["for", "range"],
    ),
    CodingTask(
        idx=7,
        name="dead_code_elim",
        difficulty="medium",
        token_budget=600,
        spec=(
            "The function below contains dead code (an unused variable and an "
            "unreachable branch). Remove the dead code while preserving the "
            "function's behavior EXACTLY. Return the cleaned function.\n\n"
            "```python\n"
            "def classify(n):\n"
            "    unused = 42\n"
            "    if n > 0:\n"
            "        return 'positive'\n"
            "    elif n == 0:\n"
            "        return 'zero'\n"
            "    else:\n"
            "        if False:\n"
            "            return 'impossible'\n"
            "        return 'negative'\n"
            "```\n"
            "The cleaned version must produce identical output for all inputs."
        ),
        signature="def classify(n: int) -> str:",
        tests=[
            "assert classify(5) == 'positive'",
            "assert classify(0) == 'zero'",
            "assert classify(-3) == 'negative'",
            "assert classify(1) == 'positive'",
            "assert classify(-1) == 'negative'",
            "assert classify(100) == 'positive'",
            "assert classify(-100) == 'negative'",
            "assert classify(0) == 'zero'",
        ],
        expected_concepts=["dead", "unused", "unreachable", "remove", "preserve", "behavior", "false"],
        required_params=["n"],
        required_constructs=["if", "return"],
    ),
    CodingTask(
        idx=8,
        name="sliding_window_max",
        difficulty="hard",
        token_budget=700,
        spec=(
            "Implement `sliding_window_max(arr, k)` returning a list of the "
            "maximum value in each sliding window of size `k` over `arr`. Must "
            "be O(n) using a deque — do NOT call `max()` on each window slice. "
            "If k > len(arr), return []. If k <= 0, return []."
        ),
        signature="def sliding_window_max(arr: list[int], k: int) -> list[int]:",
        tests=[
            "assert sliding_window_max([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7]",
            "assert sliding_window_max([1], 1) == [1]",
            "assert sliding_window_max([1,2,3,4,5], 2) == [2,3,4,5]",
            "assert sliding_window_max([5,4,3,2,1], 2) == [5,4,3,2]",
            "assert sliding_window_max([], 3) == []",
            "assert sliding_window_max([1,2,3], 5) == []",
            "assert sliding_window_max([1,2,3], 0) == []",
            "assert sliding_window_max([1,3,2,5,4], 3) == [3,5,5]",
            "assert sliding_window_max([7,7,7,7], 2) == [7,7,7]",
        ],
        expected_concepts=["deque", "window", "monotonic", "max", "o(n)", "slide", "queue"],
        required_params=["arr", "k"],
        required_constructs=["deque", "while"],
    ),
    CodingTask(
        idx=9,
        name="edge_case_handler",
        difficulty="medium",
        token_budget=500,
        spec=(
            "Implement `safe_divide(a, b)` that returns `a / b` as a float, but "
            "handles edge cases: if `b` is 0, return `float('inf')` with the "
            "same sign as `a` (positive inf for positive a, negative inf for "
            "negative a, and NaN for a==0). If either argument is not a number, "
            "raise TypeError. Do NOT let a ZeroDivisionError escape."
        ),
        signature="def safe_divide(a, b) -> float:",
        tests=[
            "assert safe_divide(10, 2) == 5.0",
            "assert safe_divide(1, 3) == 1/3",
            "assert safe_divide(10, 0) == float('inf')",
            "assert safe_divide(-10, 0) == float('-inf')",
            "import math; assert math.isnan(safe_divide(0, 0))",
            "assert safe_divide(0, 5) == 0.0",
            "assert safe_divide(-6, 3) == -2.0",
            "assert safe_divide(7, -2) == -3.5",
            "try: safe_divide('a', 2); assert False\nexcept TypeError: pass",
            "try: safe_divide(2, 'b'); assert False\nexcept TypeError: pass",
        ],
        expected_concepts=["zero", "edge", "inf", "nan", "type", "divide", "sign", "handle"],
        required_params=["a", "b"],
        required_constructs=["if", "isinstance"],
    ),
    CodingTask(
        idx=10,
        name="graph_shortest_path",
        difficulty="hard",
        token_budget=800,
        spec=(
            "Implement `shortest_path(graph, start, end)` where `graph` is an "
            "adjacency dict {node: [neighbors]}. Return the shortest path as a "
            "list of nodes from start to end (inclusive), or None if no path. "
            "Use BFS. Handle start==end (return [start]) and disconnected graphs."
        ),
        signature="def shortest_path(graph: dict, start, end) -> list | None:",
        tests=[
            "assert shortest_path({'a':['b'],'b':['c'],'c':[]}, 'a', 'c') == ['a','b','c']",
            "assert shortest_path({'a':['b','c'],'b':['d'],'c':['d'],'d':[]}, 'a', 'd') == ['a','b','d'] or shortest_path({'a':['b','c'],'b':['d'],'c':['d'],'d':[]}, 'a', 'd') == ['a','c','d']",
            "assert shortest_path({'a':['b'],'b':[]}, 'a', 'a') == ['a']",
            "assert shortest_path({'a':['b'],'b':[]}, 'a', 'c') == None",
            "assert shortest_path({'a':[]}, 'a', 'a') == ['a']",
            "assert shortest_path({'a':['b'],'b':['a']}, 'a', 'b') == ['a','b']",
            "assert shortest_path({'x':['y','z'],'y':['w'],'z':['w'],'w':[]}, 'x', 'w') == ['x','y','w'] or shortest_path({'x':['y','z'],'y':['w'],'z':['w'],'w':[]}, 'x', 'w') == ['x','z','w']",
            "g = {0:[1,2],1:[3],2:[3],3:[4],4:[]}; r = shortest_path(g,0,4); assert r == [0,1,3,4] or r == [0,2,3,4]",
        ],
        expected_concepts=["bfs", "queue", "visited", "path", "shortest", "neighbor", "breadth"],
        required_params=["graph", "start", "end"],
        required_constructs=["deque", "while"],
    ),
    CodingTask(
        idx=11,
        name="api_migration",
        difficulty="medium",
        token_budget=600,
        spec=(
            "Migrate the old API function to the new one. The old function "
            "`get_user_data_old(user_id, include_email, include_phone)` uses "
            "boolean flags. Write `get_user_data_new(user_id, fields)` where "
            "`fields` is a list of strings like `['email', 'phone']`. Return a "
            "dict with 'id' always present, plus requested fields set to sample "
            "values: email -> f'{user_id}@example.com', phone -> '555-0000'. "
            "Unknown fields are ignored. The new function must NOT accept "
            "boolean flags."
        ),
        signature="def get_user_data_new(user_id: int, fields: list[str]) -> dict:",
        tests=[
            "assert get_user_data_new(1, ['email']) == {'id':1, 'email':'1@example.com'}",
            "assert get_user_data_new(1, []) == {'id':1}",
            "assert get_user_data_new(5, ['email','phone']) == {'id':5, 'email':'5@example.com', 'phone':'555-0000'}",
            "assert get_user_data_new(3, ['phone']) == {'id':3, 'phone':'555-0000'}",
            "assert get_user_data_new(0, ['unknown']) == {'id':0}",
            "assert get_user_data_new(1, ['email','email']) == {'id':1, 'email':'1@example.com'}",
            "assert get_user_data_new(99, ['email','phone','bad']) == {'id':99, 'email':'99@example.com', 'phone':'555-0000'}",
        ],
        expected_concepts=["migrate", "fields", "api", "list", "dict", "flag", "new"],
        required_params=["user_id", "fields"],
        required_constructs=["for", "in"],
    ),
]


__all__ = ["CodingTask", "TASKS"]
