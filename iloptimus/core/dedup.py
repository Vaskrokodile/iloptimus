"""Sub-quadratic near-duplicate detection for corpus-scale dataset curation.

The session-scale filter in ``dataset_tools`` compares every candidate row's
5-token shingle set against every previously accepted row — exact, but O(n^2).
That is fine for 2k rows and fatal for 100k. This module provides the scaled
replacements while preserving the same textual semantics:

- Text is normalized exactly like ``dataset_tools._normalize``.
- Shingles are the same 5-token windows hashed with BLAKE2b (8-byte digests).
- Near-duplicate decisions compare Jaccard similarity against the same
  threshold; ``ExactJaccardGuard`` computes true Jaccard (legacy semantics),
  ``MinHashDuplicateGuard`` estimates it with MinHash signatures and uses an
  LSH band index so candidate lookup is O(bands) instead of O(rows).
- ``BloomFilter`` gives cheap probabilistic membership for global exact-dup
  pre-screening across corpus jobs.

Everything here is dependency-free and deterministic.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import re
from pathlib import Path
from typing import Iterable

# Mersenne prime used by the MinHash universal hash family.
_MERSENNE_P = (1 << 61) - 1
# Smaller Mersenne prime for the vectorized (numpy) path: products stay below
# 2^62, so int64 arithmetic never overflows.
_MERSENNE_P31 = (1 << 31) - 1
_MASK64 = (1 << 64) - 1

try:  # numpy is optional; the pure-Python fallback stays deterministic.
    import numpy as _numpy
except ImportError:  # pragma: no cover - exercised only without numpy
    _numpy = None

DEFAULT_NUM_PERM = 128
DEFAULT_THRESHOLD = 0.84
DEFAULT_SHINGLE_WIDTH = 5


def normalize_text(text: str) -> str:
    """Identical normalization to ``dataset_tools._normalize`` (single source)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_$]+", " ", text.casefold())).strip()


def _shingle_hashes(text: str, width: int = DEFAULT_SHINGLE_WIDTH) -> list[int]:
    """Hash each token window exactly like ``dataset_tools._shingles``."""
    tokens = normalize_text(text).split()
    if not tokens:
        return []
    if len(tokens) < width:
        joined = " ".join(tokens)
        return [int.from_bytes(hashlib.blake2b(joined.encode(), digest_size=8).digest(), "big")]
    return [
        int.from_bytes(
            hashlib.blake2b(" ".join(tokens[i : i + width]).encode(), digest_size=8).digest(),
            "big",
        )
        for i in range(len(tokens) - width + 1)
    ]


def shingle_set(text: str, width: int = DEFAULT_SHINGLE_WIDTH) -> set[int]:
    """The exact shingle set used by the legacy Jaccard comparison."""
    return set(_shingle_hashes(text, width))


def jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# ---------------------------------------------------------------------------
# Guards (drop-in "is this a near-duplicate of anything accepted so far?")
# ---------------------------------------------------------------------------


class ExactJaccardGuard:
    """Legacy semantics: true Jaccard against every accepted fingerprint.

    O(n^2) — used for small datasets and for equivalence testing of the
    streaming/MinHash path.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, width: int = DEFAULT_SHINGLE_WIDTH) -> None:
        self.threshold = threshold
        self.width = width
        self._fingerprints: list[set[int]] = []

    def __len__(self) -> int:
        return len(self._fingerprints)

    def is_duplicate(self, text: str) -> bool:
        fingerprint = shingle_set(text, self.width)
        if not fingerprint:
            return True
        if any(jaccard(fingerprint, prior) >= self.threshold for prior in self._fingerprints):
            return True
        self._fingerprints.append(fingerprint)
        return False


class MinHashSignature:
    """Deterministic MinHash signer over 64-bit shingle hashes."""

    def __init__(self, num_perm: int = DEFAULT_NUM_PERM, seed: int = 0x5EED) -> None:
        if num_perm <= 0 or num_perm % 2:
            raise ValueError("num_perm must be a positive even integer")
        self.num_perm = num_perm
        import random

        rng = random.Random(seed)
        self._a = [rng.randrange(1, _MERSENNE_P) for _ in range(num_perm)]
        self._b = [rng.randrange(0, _MERSENNE_P) for _ in range(num_perm)]
        # Vectorized-path coefficients reduced into the 31-bit field.
        self._a32 = [value % _MERSENNE_P31 or 1 for value in self._a]
        self._b32 = [value % _MERSENNE_P31 for value in self._b]

    def signature(self, text: str, width: int = DEFAULT_SHINGLE_WIDTH) -> list[int]:
        hashes = _shingle_hashes(text, width)
        if not hashes:
            return [_MERSENNE_P] * self.num_perm
        if _numpy is not None:
            h = _numpy.array([value % _MERSENNE_P31 for value in hashes], dtype=_numpy.int64)
            a = _numpy.array(self._a32, dtype=_numpy.int64)
            b = _numpy.array(self._b32, dtype=_numpy.int64)
            values = (a[:, None] * h[None, :] + b[:, None]) % _MERSENNE_P31
            return values.min(axis=1).tolist()
        signature = [_MERSENNE_P] * self.num_perm
        for h in hashes:
            h %= _MERSENNE_P
            for i, (a, b) in enumerate(zip(self._a, self._b)):
                value = (a * h + b) % _MERSENNE_P
                if value < signature[i]:
                    signature[i] = value
        return signature

    @staticmethod
    def estimated_jaccard(sig_a: Iterable[int], sig_b: Iterable[int]) -> float:
        a = list(sig_a)
        b = list(sig_b)
        if not a or len(a) != len(b):
            return 0.0
        equal = sum(1 for x, y in zip(a, b) if x == y)
        return equal / len(a)


class LSHIndex:
    """Banded LSH candidate index over MinHash signatures.

    Band/row counts are derived from the similarity threshold so the
    S-curve is steep near it: pairs at the threshold become candidates with
    probability ~= 1 - (1 - t^r)^b, which is ~0.99 for the default choices.
    Candidate pairs must still be confirmed with ``estimated_jaccard``.
    """

    def __init__(self, num_perm: int = DEFAULT_NUM_PERM, threshold: float = DEFAULT_THRESHOLD) -> None:
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        self.num_perm = num_perm
        self.threshold = threshold
        # Search divisor splits of num_perm for the band count whose S-curve
        # inflection is closest to the threshold.
        best = None
        for bands in range(1, num_perm + 1):
            if num_perm % bands:
                continue
            rows = num_perm // bands
            inflection = (1.0 / bands) ** (1.0 / rows)
            distance = abs(inflection - threshold)
            if best is None or distance < best[0]:
                best = (distance, bands, rows)
        _, self.bands, self.rows = best
        self._buckets: list[dict[tuple[int, ...], list[str]]] = [
            {} for _ in range(self.bands)
        ]
        self._signatures: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self._signatures)

    def add(self, key: str, signature: list[int]) -> None:
        if key in self._signatures:
            return
        self._signatures[key] = list(signature)
        for band in range(self.bands):
            band_slice = tuple(signature[band * self.rows : (band + 1) * self.rows])
            self._buckets[band].setdefault(band_slice, []).append(key)

    def candidates(self, signature: list[int]) -> list[str]:
        found: set[str] = set()
        for band in range(self.bands):
            band_slice = tuple(signature[band * self.rows : (band + 1) * self.rows])
            for key in self._buckets[band].get(band_slice, ()):
                found.add(key)
        return list(found)

    def signature_for(self, key: str) -> list[int] | None:
        return self._signatures.get(key)


class MinHashDuplicateGuard:
    """Scaled near-duplicate guard: LSH candidate lookup + signature Jaccard.

    ``is_duplicate(text)`` returns True when an already-accepted text has
    estimated Jaccard >= threshold; otherwise the text is accepted (its
    signature is stored) and False is returned.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        num_perm: int = DEFAULT_NUM_PERM,
        width: int = DEFAULT_SHINGLE_WIDTH,
        seed: int = 0x5EED,
    ) -> None:
        self.threshold = threshold
        self.width = width
        self._signer = MinHashSignature(num_perm=num_perm, seed=seed)
        self._index = LSHIndex(num_perm=num_perm, threshold=threshold)
        self._counter = 0

    def __len__(self) -> int:
        return len(self._index)

    def is_duplicate(self, text: str) -> bool:
        normalized = normalize_text(text)
        if not normalized:
            return True
        signature = self._signer.signature(text, self.width)
        for candidate_key in self._index.candidates(signature):
            prior = self._index.signature_for(candidate_key)
            if prior is not None and MinHashSignature.estimated_jaccard(signature, prior) >= self.threshold:
                return True
        self._counter += 1
        self._index.add(f"row-{self._counter}", signature)
        return False


# ---------------------------------------------------------------------------
# Bloom filter (cheap global membership pre-screen)
# ---------------------------------------------------------------------------


class BloomFilter:
    """Standard double-hashed bloom filter with deterministic BLAKE2b hashes."""

    def __init__(self, capacity: int = 1_000_000, error_rate: float = 1e-6) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 < error_rate < 1.0:
            raise ValueError("error_rate must be in (0, 1)")
        self.capacity = capacity
        self.error_rate = error_rate
        self.num_bits = max(64, math.ceil(-(capacity * math.log(error_rate)) / (math.log(2) ** 2)))
        self.num_hashes = max(1, round((self.num_bits / capacity) * math.log(2)))
        self._bits = bytearray((self.num_bits + 7) // 8)
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def _positions(self, item: str) -> list[int]:
        digest = hashlib.blake2b(item.encode(), digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:], "big")
        return [(h1 + i * h2) % self.num_bits for i in range(self.num_hashes)]

    def add(self, item: str) -> None:
        for position in self._positions(item):
            self._bits[position >> 3] |= 1 << (position & 7)
        self._count += 1

    def __contains__(self, item: str) -> bool:
        return all(
            self._bits[position >> 3] & (1 << (position & 7))
            for position in self._positions(item)
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "capacity": self.capacity,
                    "error_rate": self.error_rate,
                    "num_bits": self.num_bits,
                    "num_hashes": self.num_hashes,
                    "bits": bytes(self._bits),
                    "count": self._count,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path) -> "BloomFilter":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        bloom = cls(capacity=payload["capacity"], error_rate=payload["error_rate"])
        bloom.num_bits = payload["num_bits"]
        bloom.num_hashes = payload["num_hashes"]
        bloom._bits = bytearray(payload["bits"])
        bloom._count = payload["count"]
        return bloom
