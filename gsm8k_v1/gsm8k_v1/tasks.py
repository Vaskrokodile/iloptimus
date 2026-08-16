"""GSM8K grade-school math benchmark tasks for the IL self-improvement pipeline.

A curated subset of 25 GSM8K math word problems covering different skill areas
(arithmetic, multi-step reasoning, unit conversion, percentages, fractions,
rates, geometry, logic).

Each task has:
- a spec (the math word problem)
- a deterministic verifier that extracts the final numeric answer and compares it
- expected reasoning concepts (for reasoning-quality scoring)

The model responds in <reasoning>...</reasoning><answer>...</answer> format.
The answer is verified by extracting the last number from the answer section
and comparing it to the expected value.

Scored with IL efficiency-aware reward shaping:

    final = correctness * (0.6 + 0.4 * reasoning_quality)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GSM8KTask:
    """A single GSM8K math word problem."""
    idx: int
    name: str
    spec: str
    verify: Callable[[str], tuple[bool, str]]
    expected_concepts: list[str]
    token_budget: int = 500
    difficulty: str = "medium"
    answer_format: str = "the final number on its own line"


def _extract_answer_text(response: str) -> str:
    """Extract <answer>...</answer> or fallback to text after </reasoning>."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"</reasoning>\s*(.*)", response, re.DOTALL)
    return m.group(1).strip() if m else response.strip()


def _numeric_answer(expected: float):
    """Verifier that extracts the last number from the answer and compares to expected."""
    def verify(ans: str) -> tuple[bool, str]:
        # Extract all numbers (including decimals and negatives)
        nums = re.findall(r"-?\d+(?:\.\d+)?", ans)
        if nums:
            try:
                got = float(nums[-1])
                # Allow exact match or within 0.01 tolerance for floating point
                if abs(got - expected) < 0.01 or round(got) == round(expected):
                    return True, f"got {nums[-1]}"
            except ValueError:
                pass
        return False, f"expected {expected}, got {ans[:80]}"
    return verify


# ---------------------------------------------------------------------------
# 25 curated GSM8K tasks
# ---------------------------------------------------------------------------

TASKS: list[GSM8KTask] = [
    GSM8KTask(
        idx=0,
        name="basic_addition",
        difficulty="easy",
        token_budget=300,
        spec=(
            "Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether "
            "in April and May?"
        ),
        verify=_numeric_answer(72),
        expected_concepts=["add", "half", "sold", "total", "altogether"],
    ),
    GSM8KTask(
        idx=1,
        name="multiplication_buses",
        difficulty="easy",
        token_budget=300,
        spec=(
            "Weng earns $12 an hour for babysitting. Yesterday, she just did "
            "50 minutes of babysitting. How much did she earn?"
        ),
        verify=_numeric_answer(10),
        expected_concepts=["hour", "minute", "rate", "divide", "earn"],
    ),
    GSM8KTask(
        idx=2,
        name="percentage_books",
        difficulty="medium",
        token_budget=400,
        spec=(
            "Betty is saving money for a new wallet which costs $100. Betty has "
            "only half of the money she needs. Her parents decided to give her $15 "
            "for that purpose, and her grandparents twice as much as her parents. "
            "How much more money does Betty need to make up the remainder?"
        ),
        verify=_numeric_answer(5),
        expected_concepts=["half", "parents", "grandparents", "twice", "remainder", "save"],
    ),
    GSM8KTask(
        idx=3,
        name="multi_step_shopping",
        difficulty="medium",
        token_budget=500,
        spec=(
            "Julie is reading a 120-page book. Yesterday, she was able to read 12 "
            "pages and today, she read twice as many pages as yesterday. If she "
            "wants to read half of the remaining pages tomorrow, how many pages "
            "will she read tomorrow?"
        ),
        verify=_numeric_answer(48),
        expected_concepts=["read", "twice", "remaining", "half", "pages", "subtract"],
    ),
    GSM8KTask(
        idx=4,
        name="rate_problem",
        difficulty="medium",
        token_budget=400,
        spec=(
            "James writes a 3-page letter to 2 different friends twice a week. "
            "How many pages does he write a year?"
        ),
        verify=_numeric_answer(624),
        expected_concepts=["page", "letter", "friend", "twice", "week", "year", "multiply"],
    ),
    GSM8KTask(
        idx=5,
        name="division_groups",
        difficulty="easy",
        token_budget=300,
        spec=(
            "Mark has 36 pieces of candy and wants to share them equally among "
            "his 4 friends. How many pieces of candy will each friend receive?"
        ),
        verify=_numeric_answer(9),
        expected_concepts=["share", "equally", "divide", "candy", "friend"],
    ),
    GSM8KTask(
        idx=6,
        name="multi_step_garden",
        difficulty="medium",
        token_budget=500,
        spec=(
            "A teacher's salary is $45,000 per year. She spends $12,000 on rent, "
            "$3,000 on food, $2,000 on transportation, and $1,500 on other expenses. "
            "She saves the rest. How much does she save per year?"
        ),
        verify=_numeric_answer(26500),
        expected_concepts=["salary", "rent", "food", "transportation", "expense", "save", "subtract"],
    ),
    GSM8KTask(
        idx=7,
        name="unit_conversion",
        difficulty="medium",
        token_budget=400,
        spec=(
            "A car travels at 60 miles per hour. How many feet does the car travel "
            "in one minute? (1 mile = 5280 feet)"
        ),
        verify=_numeric_answer(5280),
        expected_concepts=["mile", "feet", "hour", "minute", "convert", "multiply", "divide"],
    ),
    GSM8KTask(
        idx=8,
        name="fraction_problem",
        difficulty="medium",
        token_budget=400,
        spec=(
            "A pizza is cut into 8 equal slices. If 3 people each eat 2 slices, "
            "how many slices are left?"
        ),
        verify=_numeric_answer(2),
        expected_concepts=["pizza", "slice", "eat", "left", "subtract", "multiply"],
    ),
    GSM8KTask(
        idx=9,
        name="age_problem",
        difficulty="medium",
        token_budget=400,
        spec=(
            "Tom is currently 12 years old. His brother Jerry is 3 years older than "
            "Tom. Their father is 25 years older than Jerry. How old is their father?"
        ),
        verify=_numeric_answer(40),
        expected_concepts=["old", "older", "brother", "father", "add", "age"],
    ),
    GSM8KTask(
        idx=10,
        name="discount_problem",
        difficulty="medium",
        token_budget=400,
        spec=(
            "A store is having a sale where all items are 20% off. If a shirt "
            "originally costs $25, how much will it cost after the discount?"
        ),
        verify=_numeric_answer(20),
        expected_concepts=["discount", "percent", "off", "original", "subtract", "multiply"],
    ),
    GSM8KTask(
        idx=11,
        name="speed_distance",
        difficulty="hard",
        token_budget=500,
        spec=(
            "A train travels 240 miles in 4 hours. At the same speed, how far "
            "will it travel in 7 hours?"
        ),
        verify=_numeric_answer(420),
        expected_concepts=["speed", "distance", "hour", "rate", "multiply", "divide"],
    ),
    GSM8KTask(
        idx=12,
        name="ratio_problem",
        difficulty="medium",
        token_budget=400,
        spec=(
            "In a class, the ratio of boys to girls is 3:2. If there are 12 boys, "
            "how many girls are there?"
        ),
        verify=_numeric_answer(8),
        expected_concepts=["ratio", "boy", "girl", "proportion", "divide", "multiply"],
    ),
    GSM8KTask(
        idx=13,
        name="area_problem",
        difficulty="medium",
        token_budget=400,
        spec=(
            "A rectangular garden is 15 feet long and 8 feet wide. What is the "
            "area of the garden in square feet?"
        ),
        verify=_numeric_answer(120),
        expected_concepts=["rectangle", "length", "width", "area", "multiply", "square"],
    ),
    GSM8KTask(
        idx=14,
        name="perimeter_problem",
        difficulty="easy",
        token_budget=300,
        spec=(
            "A square has a side length of 6 cm. What is its perimeter?"
        ),
        verify=_numeric_answer(24),
        expected_concepts=["square", "side", "perimeter", "add", "multiply"],
    ),
    GSM8KTask(
        idx=15,
        name="multi_step_recipe",
        difficulty="hard",
        token_budget=500,
        spec=(
            "A recipe requires 2 cups of flour for each batch of cookies. If you "
            "want to make 5 batches and each cup of flour weighs 4.5 ounces, how "
            "many ounces of flour do you need in total?"
        ),
        verify=_numeric_answer(45),
        expected_concepts=["recipe", "cup", "flour", "batch", "ounce", "multiply", "total"],
    ),
    GSM8KTask(
        idx=16,
        name="temperature_conversion",
        difficulty="medium",
        token_budget=400,
        spec=(
            "The temperature is 20 degrees Celsius. Convert this to Fahrenheit "
            "using the formula F = (C × 9/5) + 32."
        ),
        verify=_numeric_answer(68),
        expected_concepts=["celsius", "fahrenheit", "convert", "multiply", "divide", "add", "formula"],
    ),
    GSM8KTask(
        idx=17,
        name="average_problem",
        difficulty="medium",
        token_budget=400,
        spec=(
            "A student scored 85, 90, 78, 92, and 88 on five tests. What is the "
            "student's average (mean) score?"
        ),
        verify=_numeric_answer(86.6),
        expected_concepts=["average", "mean", "score", "sum", "divide", "add"],
    ),
    GSM8KTask(
        idx=18,
        name="profit_problem",
        difficulty="medium",
        token_budget=400,
        spec=(
            "A store buys widgets for $15 each and sells them for $22 each. If "
            "they sell 45 widgets, what is the total profit?"
        ),
        verify=_numeric_answer(315),
        expected_concepts=["buy", "sell", "profit", "widget", "subtract", "multiply"],
    ),
    GSM8KTask(
        idx=19,
        name="time_problem",
        difficulty="medium",
        token_budget=400,
        spec=(
            "A movie starts at 7:15 PM and ends at 9:45 PM. How many minutes long "
            "is the movie?"
        ),
        verify=_numeric_answer(150),
        expected_concepts=["start", "end", "hour", "minute", "subtract", "convert", "movie"],
    ),
    GSM8KTask(
        idx=20,
        name="mixture_problem",
        difficulty="hard",
        token_budget=500,
        spec=(
            "A chemist has 500 ml of a 10% acid solution. How much pure acid must "
            "be added to make it a 25% solution? Round to the nearest ml."
        ),
        verify=_numeric_answer(100),
        expected_concepts=["acid", "solution", "percent", "pure", "concentration", "algebra", "mixture"],
    ),
    GSM8KTask(
        idx=21,
        name="work_problem",
        difficulty="hard",
        token_budget=500,
        spec=(
            "If 3 workers can paint a fence in 8 hours, how long would it take "
            "4 workers to paint the same fence? (Assume all workers work at the "
            "same rate.)"
        ),
        verify=_numeric_answer(6),
        expected_concepts=["worker", "hour", "rate", "inverse", "proportion", "multiply", "divide"],
    ),
    GSM8KTask(
        idx=22,
        name="compound_interest",
        difficulty="hard",
        token_budget=500,
        spec=(
            "You deposit $1000 in a bank account that pays 5% simple interest per "
            "year. How much interest will you earn after 3 years?"
        ),
        verify=_numeric_answer(150),
        expected_concepts=["deposit", "interest", "percent", "year", "simple", "multiply"],
    ),
    GSM8KTask(
        idx=23,
        name="scaling_problem",
        difficulty="medium",
        token_budget=400,
        spec=(
            "A map has a scale where 1 inch represents 50 miles. If two cities are "
            "3.5 inches apart on the map, how many miles apart are they in reality?"
        ),
        verify=_numeric_answer(175),
        expected_concepts=["map", "scale", "inch", "mile", "represent", "multiply"],
    ),
    GSM8KTask(
        idx=24,
        name="remainder_problem",
        difficulty="medium",
        token_budget=400,
        spec=(
            "A school has 350 students. If each classroom can hold 30 students, "
            "how many students will be in the partially-filled classroom (the one "
            "that isn't full)?"
        ),
        verify=_numeric_answer(20),
        expected_concepts=["student", "classroom", "divide", "remainder", "modulo", "partial"],
    ),
]
