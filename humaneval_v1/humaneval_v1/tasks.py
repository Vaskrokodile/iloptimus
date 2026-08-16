"""HumanEval coding benchmark tasks for the IL self-improvement pipeline.

A curated subset of 25 HumanEval problems covering different difficulties and
skill areas (string manipulation, math, algorithms, data structures, edge cases).

Each task has:
- a spec (the problem description from HumanEval)
- a signature (the function signature the model must implement)
- hidden test cases (assert statements run in a sandbox)
- expected reasoning concepts (for reasoning-quality scoring)
- an anti-laziness profile (required params and constructs)

The model responds in <reasoning>...</reasoning><answer>```python\n...\n```</answer>
format. The reward runs the code in a sandbox, scores test pass-rate with an
anti-laziness penalty, then shapes the final reward by reasoning quality:

    final = correctness * (0.6 + 0.4 * reasoning_quality)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HumanEvalTask:
    """A single HumanEval coding task."""
    idx: int
    name: str
    spec: str
    signature: str
    tests: list[str]
    entry_point: str
    expected_concepts: list[str]
    token_budget: int = 600
    difficulty: str = "medium"
    required_params: list[str] = field(default_factory=list)
    required_constructs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 25 curated HumanEval tasks
# ---------------------------------------------------------------------------

TASKS: list[HumanEvalTask] = [
    HumanEvalTask(
        idx=0,
        name="has_close_elements",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a list of floating point numbers and a threshold, check if any two "
            "numbers in the list are closer to each other than the threshold. Return True "
            "if so, False otherwise."
        ),
        signature="def has_close_elements(numbers: list[float], threshold: float) -> bool:",
        entry_point="has_close_elements",
        tests=[
            "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False",
            "assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True",
            "assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False",
            "assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True",
            "assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0], 0.05) == False",
            "assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True",
            "assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False",
        ],
        expected_concepts=["pair", "distance", "abs", "threshold", "compare", "close"],
        required_params=["numbers", "threshold"],
        required_constructs=["for", "abs"],
    ),
    HumanEvalTask(
        idx=1,
        name="separate_paren_groups",
        difficulty="medium",
        token_budget=600,
        spec=(
            "Given a string of balanced parentheses, return a list of strings, each "
            "containing a separate group of balanced parentheses. Groups are separated "
            "by being adjacent but not nested."
        ),
        signature="def separate_paren_groups(paren_string: str) -> list[str]:",
        entry_point="separate_paren_groups",
        tests=[
            "assert separate_paren_groups('(()())') == ['(()())']",
            "assert separate_paren_groups('()()()') == ['()', '()', '()']",
            "assert separate_paren_groups('((()))') == ['((()))']",
            "assert separate_paren_groups('(()())(())') == ['(()())', '(())']",
            "assert separate_paren_groups('(()())(())()') == ['(()())', '(())', '()']",
            "assert separate_paren_groups('()') == ['()']",
            "assert separate_paren_groups('') == []",
        ],
        expected_concepts=["balance", "group", "paren", "count", "separate", "split"],
        required_params=["paren_string"],
        required_constructs=["for", "append"],
    ),
    HumanEvalTask(
        idx=2,
        name="truncate_number",
        difficulty="easy",
        token_budget=300,
        spec=(
            "Given a positive floating point number, return its decimal part (the part "
            "after the decimal point). For example, truncate_number(8.345) should return 0.345."
        ),
        signature="def truncate_number(number: float) -> float:",
        entry_point="truncate_number",
        tests=[
            "assert abs(truncate_number(8.5) - 0.5) < 0.001",
            "assert abs(truncate_number(1.25) - 0.25) < 0.001",
            "assert abs(truncate_number(3.5) - 0.5) < 0.001",
            "assert truncate_number(10.0) == 0.0",
            "assert abs(truncate_number(0.75) - 0.75) < 0.001",
        ],
        expected_concepts=["decimal", "float", "int", "subtract", "fractional"],
        required_params=["number"],
        required_constructs=["int", "return"],
    ),
    HumanEvalTask(
        idx=3,
        name="below_zero",
        difficulty="easy",
        token_budget=300,
        spec=(
            "Given a list of integers representing account operations, return True if "
            "the balance ever drops below zero at any point. Start with balance 0 and "
            "apply each operation in order."
        ),
        signature="def below_zero(operations: list[int]) -> bool:",
        entry_point="below_zero",
        tests=[
            "assert below_zero([1, 2, 3]) == False",
            "assert below_zero([1, 2, -4, 5]) == True",
            "assert below_zero([1, 2, -3, 1, 2]) == False",
            "assert below_zero([]) == False",
            "assert below_zero([-1]) == True",
            "assert below_zero([5, -3, -3]) == True",
        ],
        expected_concepts=["balance", "negative", "sum", "track", "zero"],
        required_params=["operations"],
        required_constructs=["for"],
    ),
    HumanEvalTask(
        idx=4,
        name="mean_absolute_deviation",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a list of numbers, return the mean absolute deviation around the mean. "
            "That is, compute the mean, then for each number compute the absolute difference "
            "from the mean, and return the average of those differences."
        ),
        signature="def mean_absolute_deviation(numbers: list[float]) -> float:",
        entry_point="mean_absolute_deviation",
        tests=[
            "assert mean_absolute_deviation([1.0, 2.0, 3.0]) == 2.0 / 3.0",
            "assert mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) == 1.0",
            "assert mean_absolute_deviation([5.0, 5.0, 5.0]) == 0.0",
            "assert mean_absolute_deviation([1.0]) == 0.0",
            "assert mean_absolute_deviation([0.0, 0.0, 0.0, 0.0]) == 0.0",
        ],
        expected_concepts=["mean", "absolute", "deviation", "average", "abs"],
        required_params=["numbers"],
        required_constructs=["sum", "abs", "len"],
    ),
    HumanEvalTask(
        idx=5,
        name="intersperse",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a list of integers and a delimiter value, return a new list where "
            "the delimiter is inserted between every two consecutive elements of the list."
        ),
        signature="def intersperse(numbers: list[int], delimiter: int) -> list[int]:",
        entry_point="intersperse",
        tests=[
            "assert intersperse([], 4) == []",
            "assert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]",
            "assert intersperse([1], 4) == [1]",
            "assert intersperse([1, 2], 7) == [1, 7, 2]",
            "assert intersperse([5, 6, 7, 8], 0) == [5, 0, 6, 0, 7, 0, 8]",
        ],
        expected_concepts=["interleave", "delimiter", "insert", "between", "alternate"],
        required_params=["numbers", "delimiter"],
        required_constructs=["for", "append"],
    ),
    HumanEvalTask(
        idx=6,
        name="parse_nested_parens",
        difficulty="medium",
        token_budget=600,
        spec=(
            "Given a string containing multiple groups of nested parentheses, return "
            "a list of integers representing the maximum nesting depth of each group. "
            "Groups are separated by spaces."
        ),
        signature="def parse_nested_parens(paren_string: str) -> list[int]:",
        entry_point="parse_nested_parens",
        tests=[
            "assert parse_nested_parens('(()())') == [2]",
            "assert parse_nested_parens('()()()') == [1, 1, 1]",
            "assert parse_nested_parens('((()))') == [3]",
            "assert parse_nested_parens('(()())(())') == [2, 2]",
            "assert parse_nested_parens('(()())(())()') == [2, 2, 1]",
            "assert parse_nested_parens('()') == [1]",
        ],
        expected_concepts=["depth", "nest", "count", "max", "paren", "balance"],
        required_params=["paren_string"],
        required_constructs=["for", "max"],
    ),
    HumanEvalTask(
        idx=7,
        name="string_xor",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given two strings of equal length consisting of '1' and '0' characters, "
            "return their XOR as a string (1 if bits differ, 0 if same)."
        ),
        signature="def string_xor(a: str, b: str) -> str:",
        entry_point="string_xor",
        tests=[
            "assert string_xor('111000', '101010') == '010010'",
            "assert string_xor('1', '1') == '0'",
            "assert string_xor('0', '0') == '0'",
            "assert string_xor('1', '0') == '1'",
            "assert string_xor('0101', '1010') == '1111'",
            "assert string_xor('0000', '0000') == '0000'",
        ],
        expected_concepts=["xor", "bit", "compare", "differ", "zip"],
        required_params=["a", "b"],
        required_constructs=["for", "zip"],
    ),
    HumanEvalTask(
        idx=8,
        name="longest",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a list of strings, return the longest string. If there are ties, "
            "return the first one. If the list is empty, return None."
        ),
        signature="def longest(strings: list[str]) -> str | None:",
        entry_point="longest",
        tests=[
            "assert longest([]) == None",
            "assert longest(['x', 'y', 'z']) == 'x'",
            "assert longest(['x', 'yyy', 'zzzz', 'www', 'oooo', 'kkkkk']) == 'kkkkk'",
            "assert longest(['a', 'bb', 'ccc']) == 'ccc'",
            "assert longest(['hello', 'world', 'hi']) == 'hello'",
        ],
        expected_concepts=["longest", "length", "max", "compare", "first"],
        required_params=["strings"],
        required_constructs=["for", "len"],
    ),
    HumanEvalTask(
        idx=9,
        name="greatest_common_divisor",
        difficulty="medium",
        token_budget=500,
        spec=(
            "Given two integers a and b, return their greatest common divisor (GCD) "
            "using the Euclidean algorithm."
        ),
        signature="def greatest_common_divisor(a: int, b: int) -> int:",
        entry_point="greatest_common_divisor",
        tests=[
            "assert greatest_common_divisor(48, 18) == 6",
            "assert greatest_common_divisor(7, 13) == 1",
            "assert greatest_common_divisor(100, 25) == 25",
            "assert greatest_common_divisor(0, 5) == 5",
            "assert greatest_common_divisor(12, 12) == 12",
            "assert greatest_common_divisor(17, 17) == 17",
        ],
        expected_concepts=["gcd", "euclidean", "modulo", "remainder", "divide"],
        required_params=["a", "b"],
        required_constructs=["while", "%"],
    ),
    HumanEvalTask(
        idx=10,
        name="all_prefixes",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a string, return a list of all its prefixes sorted from shortest to "
            "longest. The empty string is not included."
        ),
        signature="def all_prefixes(string: str) -> list[str]:",
        entry_point="all_prefixes",
        tests=[
            "assert all_prefixes('abc') == ['a', 'ab', 'abc']",
            "assert all_prefixes('a') == ['a']",
            "assert all_prefixes('hello') == ['h', 'he', 'hel', 'hell', 'hello']",
            "assert all_prefixes('xyz') == ['x', 'xy', 'xyz']",
            "assert all_prefixes('') == []",
        ],
        expected_concepts=["prefix", "slice", "substring", "range", "build"],
        required_params=["string"],
        required_constructs=["for", "append"],
    ),
    HumanEvalTask(
        idx=11,
        name="string_sequence",
        difficulty="easy",
        token_budget=300,
        spec=(
            "Given a non-negative integer n, return a string of space-separated numbers "
            "from 0 to n inclusive. For example, string_sequence(5) returns '0 1 2 3 4 5'."
        ),
        signature="def string_sequence(n: int) -> str:",
        entry_point="string_sequence",
        tests=[
            "assert string_sequence(0) == '0'",
            "assert string_sequence(5) == '0 1 2 3 4 5'",
            "assert string_sequence(3) == '0 1 2 3'",
            "assert string_sequence(10) == '0 1 2 3 4 5 6 7 8 9 10'",
            "assert string_sequence(1) == '0 1'",
        ],
        expected_concepts=["range", "join", "string", "sequence", "convert"],
        required_params=["n"],
        required_constructs=["range", "join"],
    ),
    HumanEvalTask(
        idx=12,
        name="count_distinct_characters",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a string, count the number of distinct characters (case-insensitive). "
            "For example, 'aA' has 1 distinct character."
        ),
        signature="def count_distinct_characters(string: str) -> int:",
        entry_point="count_distinct_characters",
        tests=[
            "assert count_distinct_characters('aA') == 1",
            "assert count_distinct_characters('abc') == 3",
            "assert count_distinct_characters('aAbBcC') == 3",
            "assert count_distinct_characters('') == 0",
            "assert count_distinct_characters('Hello World') == 8",
            "assert count_distinct_characters('aaa') == 1",
        ],
        expected_concepts=["distinct", "unique", "case", "lower", "set", "count"],
        required_params=["string"],
        required_constructs=["set", "lower"],
    ),
    HumanEvalTask(
        idx=13,
        name="parse_music",
        difficulty="medium",
        token_budget=600,
        spec=(
            "Given a string of musical notes, return a list of integers representing "
            "the number of beats for each note. Notes are separated by spaces. "
            "'o' = 4 beats, 'o|' = 2 beats, '.|' = 1 beat."
        ),
        signature="def parse_music(music_string: str) -> list[int]:",
        entry_point="parse_music",
        tests=[
            "assert parse_music('') == []",
            "assert parse_music('o o| .| o| o| .| .| .| .| o o') == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]",
            "assert parse_music('o| o| .| .| o o o .|') == [2, 2, 1, 1, 4, 4, 4, 1]",
            "assert parse_music('o') == [4]",
            "assert parse_music('.| .| .|') == [1, 1, 1]",
        ],
        expected_concepts=["note", "beat", "map", "split", "lookup", "parse"],
        required_params=["music_string"],
        required_constructs=["for", "split"],
    ),
    HumanEvalTask(
        idx=14,
        name="sum_product",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a list of integers, return a tuple of (sum, product) where sum is "
            "the sum of all elements and product is the product of all elements. "
            "Empty list returns (0, 1)."
        ),
        signature="def sum_product(numbers: list[int]) -> tuple[int, int]:",
        entry_point="sum_product",
        tests=[
            "assert sum_product([]) == (0, 1)",
            "assert sum_product([1, 2, 3, 4]) == (10, 24)",
            "assert sum_product([5]) == (5, 5)",
            "assert sum_product([1, 1, 1, 1]) == (4, 1)",
            "assert sum_product([0, 1, 2]) == (3, 0)",
        ],
        expected_concepts=["sum", "product", "tuple", "accumulate", "multiply"],
        required_params=["numbers"],
        required_constructs=["for"],
    ),
    HumanEvalTask(
        idx=15,
        name="rolling_max",
        difficulty="medium",
        token_budget=500,
        spec=(
            "Given a list of numbers, return a list of the running maximum at each "
            "position. That is, element i is the maximum of elements 0 through i."
        ),
        signature="def rolling_max(numbers: list[int]) -> list[int]:",
        entry_point="rolling_max",
        tests=[
            "assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]",
            "assert rolling_max([5, 4, 3, 2, 1]) == [5, 5, 5, 5, 5]",
            "assert rolling_max([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]",
            "assert rolling_max([]) == []",
            "assert rolling_max([7]) == [7]",
            "assert rolling_max([3, 1, 4, 1, 5, 9, 2, 6]) == [3, 3, 4, 4, 5, 9, 9, 9]",
        ],
        expected_concepts=["running", "maximum", "track", "max", "cumulative"],
        required_params=["numbers"],
        required_constructs=["for", "max"],
    ),
    HumanEvalTask(
        idx=16,
        name="is_palindrome",
        difficulty="easy",
        token_budget=300,
        spec=(
            "Given a string, return True if it is a palindrome (reads the same forwards "
            "and backwards), False otherwise."
        ),
        signature="def is_palindrome(text: str) -> bool:",
        entry_point="is_palindrome",
        tests=[
            "assert is_palindrome('racecar') == True",
            "assert is_palindrome('hello') == False",
            "assert is_palindrome('') == True",
            "assert is_palindrome('a') == True",
            "assert is_palindrome('abba') == True",
            "assert is_palindrome('abc') == False",
            "assert is_palindrome('madam') == True",
        ],
        expected_concepts=["palindrome", "reverse", "compare", "backward", "forward"],
        required_params=["text"],
        required_constructs=["return"],
    ),
    HumanEvalTask(
        idx=17,
        name="fibonacci",
        difficulty="medium",
        token_budget=500,
        spec=(
            "Given a non-negative integer n, return the n-th Fibonacci number. "
            "F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2)."
        ),
        signature="def fibonacci(n: int) -> int:",
        entry_point="fibonacci",
        tests=[
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(2) == 1",
            "assert fibonacci(3) == 2",
            "assert fibonacci(10) == 55",
            "assert fibonacci(20) == 6765",
            "assert fibonacci(7) == 13",
        ],
        expected_concepts=["fibonacci", "recursive", "sequence", "add", "base case"],
        required_params=["n"],
        required_constructs=["for", "return"],
    ),
    HumanEvalTask(
        idx=18,
        name="factorial",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a non-negative integer n, return n! (n factorial). "
            "0! = 1, n! = n * (n-1) * ... * 1."
        ),
        signature="def factorial(n: int) -> int:",
        entry_point="factorial",
        tests=[
            "assert factorial(0) == 1",
            "assert factorial(1) == 1",
            "assert factorial(5) == 120",
            "assert factorial(10) == 3628800",
            "assert factorial(3) == 6",
            "assert factorial(7) == 5040",
        ],
        expected_concepts=["factorial", "multiply", "product", "range", "accumulate"],
        required_params=["n"],
        required_constructs=["for", "range"],
    ),
    HumanEvalTask(
        idx=19,
        name="remove_duplicates",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a list of integers, return a new list with duplicates removed, "
            "preserving the order of first occurrence."
        ),
        signature="def remove_duplicates(numbers: list[int]) -> list[int]:",
        entry_point="remove_duplicates",
        tests=[
            "assert remove_duplicates([1, 2, 3, 2, 1, 4]) == [1, 2, 3, 4]",
            "assert remove_duplicates([]) == []",
            "assert remove_duplicates([1, 1, 1]) == [1]",
            "assert remove_duplicates([5, 3, 5, 3, 5]) == [5, 3]",
            "assert remove_duplicates([1, 2, 3]) == [1, 2, 3]",
        ],
        expected_concepts=["duplicate", "unique", "seen", "order", "preserve"],
        required_params=["numbers"],
        required_constructs=["for", "append"],
    ),
    HumanEvalTask(
        idx=20,
        name="flip_case",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a string, return a new string where uppercase letters become lowercase "
            "and lowercase letters become uppercase. Non-letter characters stay the same."
        ),
        signature="def flip_case(string: str) -> str:",
        entry_point="flip_case",
        tests=[
            "assert flip_case('Hello') == 'hELLO'",
            "assert flip_case('WORLD') == 'world'",
            "assert flip_case('python') == 'PYTHON'",
            "assert flip_case('') == ''",
            "assert flip_case('AbCdEf') == 'aBcDeF'",
            "assert flip_case('123!@#') == '123!@#'",
        ],
        expected_concepts=["swap", "case", "upper", "lower", "flip", "toggle"],
        required_params=["string"],
        required_constructs=["for"],
    ),
    HumanEvalTask(
        idx=21,
        name="sort_by_length",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a list of strings, sort them by length from shortest to longest. "
            "Strings of the same length should keep their original relative order "
            "(stable sort)."
        ),
        signature="def sort_by_length(strings: list[str]) -> list[str]:",
        entry_point="sort_by_length",
        tests=[
            "assert sort_by_length(['aaa', 'b', 'cc']) == ['b', 'cc', 'aaa']",
            "assert sort_by_length(['hello', 'hi', 'hey']) == ['hi', 'hey', 'hello']",
            "assert sort_by_length([]) == []",
            "assert sort_by_length(['a']) == ['a']",
            "assert sort_by_length(['ab', 'cd', 'ef']) == ['ab', 'cd', 'ef']",
        ],
        expected_concepts=["sort", "length", "stable", "order", "key"],
        required_params=["strings"],
        required_constructs=["sorted", "len"],
    ),
    HumanEvalTask(
        idx=22,
        name="reverse_string",
        difficulty="easy",
        token_budget=300,
        spec=(
            "Given a string, return its reverse. For example, "
            "reverse_string('hello') returns 'olleh'."
        ),
        signature="def reverse_string(s: str) -> str:",
        entry_point="reverse_string",
        tests=[
            "assert reverse_string('hello') == 'olleh'",
            "assert reverse_string('') == ''",
            "assert reverse_string('a') == 'a'",
            "assert reverse_string('abcdef') == 'fedcba'",
            "assert reverse_string('racecar') == 'racecar'",
        ],
        expected_concepts=["reverse", "backward", "slice", "flip", "string"],
        required_params=["s"],
        required_constructs=["return"],
    ),
    HumanEvalTask(
        idx=23,
        name="count_vowels",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a string, count the number of vowels (a, e, i, o, u) in it, "
            "case-insensitive."
        ),
        signature="def count_vowels(text: str) -> int:",
        entry_point="count_vowels",
        tests=[
            "assert count_vowels('hello') == 2",
            "assert count_vowels('aeiou') == 5",
            "assert count_vowels('xyz') == 0",
            "assert count_vowels('') == 0",
            "assert count_vowels('HELLO') == 2",
            "assert count_vowels('The quick brown fox') == 5",
        ],
        expected_concepts=["vowel", "count", "case", "lower", "check"],
        required_params=["text"],
        required_constructs=["for", "in"],
    ),
    HumanEvalTask(
        idx=24,
        name="binary_to_decimal",
        difficulty="easy",
        token_budget=400,
        spec=(
            "Given a string representing a binary number (containing only '0' and '1' "
            "characters), return its decimal integer value."
        ),
        signature="def binary_to_decimal(binary: str) -> int:",
        entry_point="binary_to_decimal",
        tests=[
            "assert binary_to_decimal('0') == 0",
            "assert binary_to_decimal('1') == 1",
            "assert binary_to_decimal('10') == 2",
            "assert binary_to_decimal('1010') == 10",
            "assert binary_to_decimal('1111') == 15",
            "assert binary_to_decimal('100000') == 32",
            "assert binary_to_decimal('11111111') == 255",
        ],
        expected_concepts=["binary", "decimal", "convert", "power", "base", "bit"],
        required_params=["binary"],
        required_constructs=["for", "int"],
    ),
]
