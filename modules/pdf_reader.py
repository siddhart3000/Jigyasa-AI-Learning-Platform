from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable

import pdfplumber


LIBRARY_ROOT = Path("pdf_library")


@dataclass(frozen=True)
class PdfDoc:
    subject: str
    path: Path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\- ]+", "", (value or "").strip())


def list_classes() -> list[str]:
    # Class hierarchy removed; kept for compatibility with untouched pages.
    return []


def list_subjects(class_name: str | None = None) -> list[str]:
    # class_name is ignored; storage is subject -> PDFs.
    if not LIBRARY_ROOT.exists():
        return []
    return sorted([p.name for p in LIBRARY_ROOT.iterdir() if p.is_dir()])


def list_pdfs(subject_or_class: str, subject: str | None = None) -> list[PdfDoc]:
    # Accept both new signature list_pdfs(subject) and old list_pdfs(class, subject).
    subject_name = subject if subject is not None else subject_or_class
    subject_dir = LIBRARY_ROOT / _safe_name(subject_name)
    if not subject_dir.exists():
        return []
    pdfs = sorted(subject_dir.glob("*.pdf"), key=lambda p: p.name.lower())
    return [PdfDoc(subject=subject_dir.name, path=p) for p in pdfs]


def create_class(class_name: str, default_subject: str | None = "English") -> Path:
    # Class hierarchy removed; kept for compatibility only.
    if default_subject:
        return create_subject(class_name, default_subject)
    LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    return LIBRARY_ROOT


def create_subject(subject_or_class: str, subject: str | None = None) -> Path:
    # Accept both create_subject(subject) and old create_subject(class, subject).
    subject_name = _safe_name(subject if subject is not None else subject_or_class)
    if not subject_name:
        raise ValueError("Invalid subject name.")
    subject_dir = LIBRARY_ROOT / subject_name
    subject_dir.mkdir(parents=True, exist_ok=True)
    return subject_dir


def save_uploaded_pdf(
    subject_or_class: str,
    subject_or_filename: str,
    filename_or_content: str | bytes,
    content: bytes | None = None,
) -> Path:
    # Accept both new save_uploaded_pdf(subject, filename, content)
    # and old save_uploaded_pdf(class, subject, filename, content).
    if content is None:
        subject_name = subject_or_class
        filename = str(subject_or_filename)
        file_bytes = filename_or_content if isinstance(filename_or_content, bytes) else b""
    else:
        subject_name = subject_or_filename
        filename = str(filename_or_content)
        file_bytes = content

    subject_dir = create_subject(subject_name)
    safe_name = Path(filename).name
    out_path = subject_dir / safe_name
    out_path.write_bytes(file_bytes)
    return out_path


def format_class_display(class_name: str) -> str:
    return _safe_name(class_name).replace("_", " ")


def format_subject_display(subject: str) -> str:
    return _safe_name(subject).replace("_", " ").title()


def format_pdf_display(filename: str) -> str:
    stem = Path(filename).stem
    return stem.replace("_", " ").strip().title()


def extract_text_from_pdf(path: Path, max_chars: int = 120_000) -> str:
    text_parts: list[str] = []
    total = 0
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if not t.strip():
                continue
            remaining = max_chars - total
            if remaining <= 0:
                break
            chunk = t[:remaining]
            text_parts.append(chunk)
            total += len(chunk)
    return "\n\n".join(text_parts).strip()


def build_context(docs: Iterable[Path], max_chars: int = 24_000) -> str:
    # Small context window for LLM prompt.
    parts: list[str] = []
    used = 0
    for p in docs:
        t = extract_text_from_pdf(p, max_chars=max_chars)
        if not t:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        snippet = t[:remaining]
        parts.append(f"[Source: {p.name}]\n{snippet}")
        used += len(snippet)
    return "\n\n".join(parts).strip()

