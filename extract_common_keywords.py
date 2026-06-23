#!/usr/bin/env python3
"""Extract common key terms from multiple PDF files.

The script reads the text of each input PDF, identifies candidate words and
short phrases, and prints up to nine terms that occur across the highest number
of PDFs. It prefers terms shared by all documents, but still returns useful
near-common terms when the input PDFs do not have nine terms in common.

Dependencies:
    pip install pypdf

Example:
    python extract_common_keywords.py report1.pdf report2.pdf report3.pdf
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - exercised when dependency is absent
    PdfReader = None  # type: ignore[assignment]


DEFAULT_STOPWORDS = {
    "aber",
    "alle",
    "allem",
    "allen",
    "aller",
    "alles",
    "als",
    "also",
    "am",
    "an",
    "and",
    "auch",
    "auf",
    "aus",
    "bei",
    "bis",
    "by",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "dies",
    "diese",
    "diesem",
    "diesen",
    "dieser",
    "dieses",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "for",
    "from",
    "für",
    "hat",
    "have",
    "im",
    "in",
    "ist",
    "it",
    "mit",
    "nicht",
    "of",
    "on",
    "oder",
    "sich",
    "sind",
    "the",
    "to",
    "und",
    "von",
    "was",
    "werden",
    "wie",
    "with",
    "zu",
    "zum",
    "zur",
}

WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ-]{2,}")


def read_pdf_text(pdf_path: Path) -> str:
    """Return extracted text from a PDF file."""
    if PdfReader is None:
        raise RuntimeError(
            "Missing dependency 'pypdf'. Install it with: python -m pip install pypdf"
        )

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 - show path-specific context to users
        raise RuntimeError(f"Could not open PDF '{pdf_path}': {exc}") from exc

    page_texts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_texts.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - continue when only one page fails
            print(
                f"Warning: could not extract page {page_number} from '{pdf_path}': {exc}",
                file=sys.stderr,
            )
    return "\n".join(page_texts)


def normalize_tokens(text: str, stopwords: set[str]) -> list[str]:
    """Tokenize text and remove common filler words."""
    tokens = [match.group(0).lower().strip("-") for match in WORD_PATTERN.finditer(text)]
    return [token for token in tokens if token and token not in stopwords]


def candidate_terms(tokens: list[str], max_ngram: int) -> Counter[str]:
    """Create a frequency counter for words and short phrases."""
    terms: Counter[str] = Counter(tokens)
    for ngram_size in range(2, max_ngram + 1):
        for start in range(0, len(tokens) - ngram_size + 1):
            phrase = " ".join(tokens[start : start + ngram_size])
            terms[phrase] += 1
    return terms


def score_common_terms(
    document_term_counts: list[Counter[str]], limit: int
) -> list[tuple[str, int, int, float]]:
    """Rank terms by document coverage, total frequency, and phrase specificity."""
    document_frequency: defaultdict[str, int] = defaultdict(int)
    total_frequency: Counter[str] = Counter()

    for term_counts in document_term_counts:
        total_frequency.update(term_counts)
        for term in term_counts:
            document_frequency[term] += 1

    document_count = len(document_term_counts)
    scored_terms: list[tuple[str, int, int, float]] = []
    for term, doc_freq in document_frequency.items():
        total_freq = total_frequency[term]
        phrase_bonus = 1.0 + 0.15 * (len(term.split()) - 1)
        coverage_bonus = doc_freq / document_count
        score = coverage_bonus * math.log1p(total_freq) * phrase_bonus
        scored_terms.append((term, doc_freq, total_freq, score))

    scored_terms.sort(key=lambda item: (item[1], item[3], item[2], len(item[0])), reverse=True)
    return scored_terms[:limit]


def extract_common_keywords(
    pdf_paths: Iterable[Path],
    limit: int = 9,
    max_ngram: int = 2,
    stopwords: set[str] | None = None,
) -> list[tuple[str, int, int, float]]:
    """Extract up to ``limit`` common key terms from the given PDFs."""
    active_stopwords = stopwords or DEFAULT_STOPWORDS
    document_term_counts: list[Counter[str]] = []

    for pdf_path in pdf_paths:
        text = read_pdf_text(pdf_path)
        tokens = normalize_tokens(text, active_stopwords)
        if not tokens:
            print(f"Warning: no extractable text found in '{pdf_path}'", file=sys.stderr)
        document_term_counts.append(candidate_terms(tokens, max_ngram))

    if not document_term_counts:
        return []
    return score_common_terms(document_term_counts, limit)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract up to nine key terms shared across multiple PDF files."
    )
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF files to analyze")
    parser.add_argument(
        "--limit",
        type=int,
        default=9,
        help="maximum number of terms to print (default: 9)",
    )
    parser.add_argument(
        "--max-ngram",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="maximum phrase length to consider (default: 2)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.limit < 1:
        print("Error: --limit must be at least 1", file=sys.stderr)
        return 2

    missing_files = [str(path) for path in args.pdfs if not path.is_file()]
    if missing_files:
        print(f"Error: PDF file(s) not found: {', '.join(missing_files)}", file=sys.stderr)
        return 2

    try:
        terms = extract_common_keywords(args.pdfs, limit=args.limit, max_ngram=args.max_ngram)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for index, (term, doc_freq, total_freq, score) in enumerate(terms, start=1):
        print(
            f"{index}. {term} "
            f"(PDFs: {doc_freq}/{len(args.pdfs)}, "
            f"Häufigkeit: {total_freq}, Score: {score:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
