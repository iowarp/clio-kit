"""Structure-aware chunking: sections, captions, tables, table cells, and equations.

Chunks carry ``structure.*``, ``table.*``, ``caption.*`` and ``equation.*`` metadata
plus a ``citation.fragment`` for table cells so citations can point at one cell.
Unit/measurement and formula semantics live in clio-converter, not here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from clio_agentic_search.models.contracts import ChunkRecord

_HEADING_PATTERN = re.compile(r"(?m)^(#{1,6})\s+(.+)$")
_CAPTION_PATTERN = re.compile(r"(?mi)^(figure|fig\.?|table)\s*([0-9A-Za-z-]*)\s*[:.-]\s*(.+)$")
_BLOCK_EQUATION_PATTERN = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
_INLINE_EQUATION_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", flags=re.DOTALL)
_TABLE_LINE_PATTERN = re.compile(r"^\s*\|.*\|\s*$")


@dataclass(frozen=True, slots=True)
class StructuredChunk:
    kind: str
    text: str
    start_offset: int
    end_offset: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    chunks: list[ChunkRecord]
    metadata_by_chunk_id: dict[str, dict[str, str]]


def build_structure_aware_chunk_plan(
    *,
    namespace: str,
    document_id: str,
    text: str,
    chunk_size: int,
) -> ChunkPlan:
    if not text:
        return ChunkPlan(chunks=[], metadata_by_chunk_id={})

    structured: list[StructuredChunk] = []
    structured.extend(_build_section_chunks(text))
    structured.extend(_build_caption_chunks(text))
    structured.extend(_build_equation_chunks(text))
    structured.extend(_build_table_chunks(text))

    if not structured:
        structured.extend(_build_plain_chunks(text=text, chunk_size=chunk_size))

    structured.sort(key=lambda item: (item.start_offset, item.end_offset, item.kind))

    chunk_records: list[ChunkRecord] = []
    metadata_by_chunk_id: dict[str, dict[str, str]] = {}
    for index, item in enumerate(structured):
        chunk_id = hashlib.sha1(
            f"{namespace}:{document_id}:{item.kind}:{index}:{item.start_offset}:{item.end_offset}".encode()
        ).hexdigest()
        chunk_metadata = dict(item.metadata)
        chunk_metadata["structure.kind"] = item.kind
        chunk_records.append(
            ChunkRecord(
                namespace=namespace,
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=index,
                text=item.text,
                start_offset=item.start_offset,
                end_offset=item.end_offset,
            )
        )
        metadata_by_chunk_id[chunk_id] = chunk_metadata

    return ChunkPlan(chunks=chunk_records, metadata_by_chunk_id=metadata_by_chunk_id)


def _build_plain_chunks(*, text: str, chunk_size: int) -> list[StructuredChunk]:
    chunks: list[StructuredChunk] = []
    for start in range(0, len(text), chunk_size):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue
        chunks.append(
            StructuredChunk(kind="text", text=chunk_text, start_offset=start, end_offset=end)
        )
    return chunks


def _build_section_chunks(text: str) -> list[StructuredChunk]:
    matches = list(_HEADING_PATTERN.finditer(text))
    if not matches:
        return []

    chunks: list[StructuredChunk] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        chunks.append(
            StructuredChunk(
                kind="section",
                text=section_text,
                start_offset=start,
                end_offset=end,
                metadata={
                    "structure.section": match.group(2).strip(),
                    "structure.section_level": str(len(match.group(1))),
                },
            )
        )
    return chunks


def _build_caption_chunks(text: str) -> list[StructuredChunk]:
    chunks: list[StructuredChunk] = []
    for index, match in enumerate(_CAPTION_PATTERN.finditer(text), start=1):
        caption_text = match.group(0).strip()
        if not caption_text:
            continue
        chunks.append(
            StructuredChunk(
                kind="caption",
                text=caption_text,
                start_offset=match.start(),
                end_offset=match.end(),
                metadata={
                    "caption.label": match.group(1).lower().rstrip("."),
                    "caption.index": match.group(2).strip() or str(index),
                    "caption.body": match.group(3).strip(),
                },
            )
        )
    return chunks


def _build_equation_chunks(text: str) -> list[StructuredChunk]:
    chunks: list[StructuredChunk] = []
    blocked_ranges: list[tuple[int, int]] = []

    for match in _BLOCK_EQUATION_PATTERN.finditer(text):
        expression = match.group(1).strip()
        if not expression:
            continue
        blocked_ranges.append((match.start(), match.end()))
        chunks.append(
            StructuredChunk(
                kind="equation",
                text=expression,
                start_offset=match.start(),
                end_offset=match.end(),
                metadata={"equation.block": "true"},
            )
        )

    for match in _INLINE_EQUATION_PATTERN.finditer(text):
        if any(start <= match.start() < end for start, end in blocked_ranges):
            continue
        expression = match.group(1).strip()
        if not expression:
            continue
        chunks.append(
            StructuredChunk(
                kind="equation",
                text=expression,
                start_offset=match.start(),
                end_offset=match.end(),
                metadata={"equation.block": "false"},
            )
        )

    return chunks


def _build_table_chunks(text: str) -> list[StructuredChunk]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    offsets: list[int] = []
    total = 0
    for line in lines:
        offsets.append(total)
        total += len(line)

    chunks: list[StructuredChunk] = []
    line_index = 0
    table_index = 0
    while line_index < len(lines):
        if not _TABLE_LINE_PATTERN.match(lines[line_index]):
            line_index += 1
            continue

        start_line = line_index
        block_lines: list[str] = []
        while line_index < len(lines) and _TABLE_LINE_PATTERN.match(lines[line_index]):
            block_lines.append(lines[line_index])
            line_index += 1

        if len(block_lines) < 2 or not _is_table_separator(block_lines[1]):
            continue

        table_index += 1
        start_offset = offsets[start_line]
        end_offset = offsets[start_line] + sum(len(line) for line in block_lines)
        table_text = "".join(block_lines).strip()
        if table_text:
            chunks.append(
                StructuredChunk(
                    kind="table",
                    text=table_text,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    metadata={"table.index": str(table_index)},
                )
            )

        headers = _split_table_row(block_lines[0])
        for row_idx, row_line in enumerate(block_lines[2:], start=1):
            for col_idx, value in enumerate(_split_table_row(row_line), start=1):
                normalized_value = value.strip()
                if not normalized_value:
                    continue
                header = headers[col_idx - 1].strip() if col_idx - 1 < len(headers) else ""
                header_label = header or f"column_{col_idx}"
                chunks.append(
                    StructuredChunk(
                        kind="table_cell",
                        text=f"{header_label}: {normalized_value}",
                        start_offset=start_offset,
                        end_offset=end_offset,
                        metadata={
                            "table.index": str(table_index),
                            "table.row": str(row_idx),
                            "table.column": str(col_idx),
                            "table.column_name": header_label,
                            "citation.fragment": (
                                f"table={table_index}&row={row_idx}&column={col_idx}"
                            ),
                        },
                    )
                )

    return chunks


def _is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in _split_table_row(line)]
    if not cells:
        return False
    return all(cell and all(character in "-:" for character in cell) for cell in cells)


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]
