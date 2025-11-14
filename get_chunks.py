import os
import json
from pathlib import Path
from typing import List, Dict

from synthetic_data_kit.utils.config import (
    load_config,
    get_generation_config,
    get_path_config,
)
from synthetic_data_kit.parsers.docx_parser import DOCXParser
from synthetic_data_kit.utils.text import split_into_chunks


def collect_docx_files(input_directory: str) -> List[Path]:
    """Recursively collect all .docx files under the input directory."""
    base_path = Path(input_directory)
    return sorted(base_path.rglob("*.docx"))


def read_docx_text(file_path: Path) -> str:
    """Parse a DOCX file and return its extracted text as a single string."""
    parser = DOCXParser()
    parsed = parser.parse(str(file_path))

    # Normalize parser outputs to a single text string
    if isinstance(parsed, list):
        texts: List[str] = []
        for item in parsed:
            if isinstance(item, dict) and "text" in item and item["text"]:
                texts.append(item["text"])
        return "\n\n".join(texts)
    if isinstance(parsed, dict) and "text" in parsed:
        return str(parsed["text"]) if parsed["text"] is not None else ""
    return str(parsed) if parsed is not None else ""


def chunk_file_text(file_path: Path, chunk_size: int, overlap: int) -> List[Dict[str, str]]:
    """Split a file's text into chunks and return a list of {id, text} dicts."""
    text = read_docx_text(file_path)
    if not text.strip():
        return []

    chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    base_id = file_path.stem
    return [{"id": f"{base_id}-{index+1}", "text": chunk} for index, chunk in enumerate(chunks)]


def main(input_directory: [Path]) -> None:
    # Load configuration
    config = load_config()
    generation_config = get_generation_config(config)
    chunk_size = int(generation_config.get("chunk_size", 4000))
    overlap = int(generation_config.get("overlap", 200))

    # Resolve input directory from config
    # input_directory = get_path_config(config, "input")

    # Choose an output file path; default to data/output/docx_chunks.json
    output_root = config.get("paths", {}).get("output", {}).get("default")
    if not output_root:
        output_root = os.path.join("data", "output")
    output_path = os.path.join(output_root, "docx_chunks.json")
    os.makedirs(output_root, exist_ok=True)

    # Process documents
    docx_files = collect_docx_files(input_directory)

    all_chunks: List[Dict[str, str]] = []
    for file_path in docx_files:
        file_chunks = chunk_file_text(file_path, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(file_chunks)

    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(docx_files)} DOCX files from '{input_directory}'.")
    print(f"Wrote {len(all_chunks)} chunks to '{output_path}'.")


if __name__ == "__main__":
    main(input_directory="data/input/round2")

