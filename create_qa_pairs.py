import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from synthetic_data_kit.utils.config import load_config, get_openai_config, get_generation_config, get_prompt


def _load_chunks(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Chunks JSON must be a list of objects with 'id' and 'text'")
    return data


def _init_openai_client(api_base: Optional[str], api_key: Optional[str]):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("The 'openai' package is required. Install with 'pip install openai>=1.0.0'.") from e

    client_kwargs: Dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if api_base:
        client_kwargs["base_url"] = api_base
    return OpenAI(**client_kwargs)


def _build_messages(prompt_template: str, chunk_text: str, num_pairs: int) -> List[Dict[str, str]]:
    # Replace only the {text} placeholder to avoid interfering with other braces in the prompt
    filled_prompt = prompt_template.replace("{text}", chunk_text).replace("{num_pairs}", str(num_pairs))
    return [
        {"role": "user", "content": filled_prompt}
    ]


def _parse_qa_response(raw_content: str) -> Any:
    try:
        return json.loads(raw_content)
    except Exception:
        return raw_content  # Fallback to raw text if not valid JSON


def generate_qa_pairs_from_chunks(
    input_json_path: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
) -> Path:
    # Load configuration
    config = load_config()
    openai_cfg = get_openai_config(config)
    gen_cfg = get_generation_config(config)
    prompt_template = get_prompt(config, "qa_generation")

    # Resolve paths
    input_path = Path(input_json_path) if input_json_path else Path("data/output/docx_chunks.json")
    if output_json_path is None:
        # Derive output path from input name
        out_name = f"{input_path.stem}_qa_pairs.json"
        output_path = Path("data/generated") / out_name
    else:
        output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # API params with env overrides
    env_api_key = os.environ.get("API_ENDPOINT_KEY") or os.environ.get("OPENAI_API_KEY")
    api_key = env_api_key or openai_cfg.get("api_key")
    api_base = os.environ.get("OPENAI_BASE_URL") or openai_cfg.get("api_base")
    model_name = os.environ.get("OPENAI_MODEL") or openai_cfg.get("model")

    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.95))
    max_tokens = int(gen_cfg.get("max_tokens", 4096))
    num_pairs = int(os.environ.get("QA_NUM_PAIRS", gen_cfg.get("num_pairs", 15)))

    # Initialize client
    client = _init_openai_client(api_base=api_base, api_key=api_key)

    # Load chunks
    chunks = _load_chunks(input_path)

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(chunks, start=1):
        chunk_id = item.get("id") or f"chunk-{idx}"
        chunk_text = item.get("text", "")
        if not chunk_text:
            results.append({"id": chunk_id, "error": "Empty text"})
            continue

        messages = _build_messages(prompt_template, chunk_text, num_pairs)

        # Perform completion (OpenAI-compatible)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_object"
            },
            top_p=top_p,
            max_tokens=max_tokens,
        )

        # Try to extract content in a robust way
        content: Optional[str] = None
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message and hasattr(choice.message, "content"):
                content = choice.message.content

        # Fallback to dict-like access
        if content is None:
            try:
                response_dict = response.model_dump() if hasattr(response, "model_dump") else getattr(response, "__dict__", None)
                if response_dict and isinstance(response_dict, dict):
                    if "choices" in response_dict and response_dict["choices"]:
                        first = response_dict["choices"][0]
                        if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict):
                            content = first["message"].get("content")
            except Exception:
                content = None

        raw = content if content is not None else ""
        parsed = _parse_qa_response(raw)
        results.append({
            "id": chunk_id,
            "prompt": messages[0]["content"],
            "raw": raw,
            "qa_pairs": parsed,
        })

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote QA pairs to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_qa_pairs_from_chunks(
        "data/output/docx_chunks.json",
        "data/generated/docx_qa_pairs_chapter3-8.json"
    )


