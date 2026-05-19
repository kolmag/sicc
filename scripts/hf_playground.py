"""
Practice with Hugging Face Inference Providers using the OpenAI-compatible API.

Examples:
    uv run python scripts/hf_playground.py --challenge supplier-risk
    uv run python scripts/hf_playground.py --prompt "Explain PPAP Level 3 in 5 bullets"
    uv run python scripts/hf_playground.py --models google/gemma-2-2b-it openai/gpt-oss-20b

Requires:
    HUGGINGFACE_API_KEY, HF_TOKEN, or HUGGINGFACEHUB_API_TOKEN with Inference
    Providers permission.
"""

from __future__ import annotations

import argparse
import os
import textwrap
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HF_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODELS = [
    "Qwen/Qwen3-4B-Thinking-2507",
]


@dataclass(frozen=True)
class Challenge:
    title: str
    system: str
    prompt: str


CHALLENGES = {
    "supplier-risk": Challenge(
        title="Supplier risk analyst",
        system=(
            "You are a concise supplier quality analyst. Use practical language, "
            "separate facts from assumptions, and avoid inventing policy."
        ),
        prompt=(
            "A strategic single-source electronics supplier has OTD falling from 96% "
            "to 82% over three months, PPM rising from 220 to 980, two open SCARs "
            "older than 45 days, and an APQP launch due in six weeks. Draft a brief "
            "risk note with: risk level, likely causes to investigate, immediate "
            "containment, and what evidence you would request."
        ),
    ),
    "scar-triage": Challenge(
        title="SCAR triage",
        system=(
            "You are a supplier quality engineer triaging corrective action. "
            "Be specific, operational, and grounded in the facts provided."
        ),
        prompt=(
            "A customer line stop was traced to burrs on a machined bracket. "
            "The supplier says the deburring operator was absent and offers to "
            "retrain the backup operator. Create a SCAR response critique and a "
            "better corrective action plan."
        ),
    ),
    "prompt-doctor": Challenge(
        title="Prompt doctor",
        system=(
            "You are a prompt coach. Improve prompts for smaller open models. "
            "Prefer crisp constraints and visible output structure."
        ),
        prompt=(
            "Rewrite this vague prompt for a small free Hugging Face model: "
            "'Tell me if this supplier is bad and what to do.' Include the revised "
            "prompt and explain why it will work better."
        ),
    ),
    "red-team": Challenge(
        title="RAG red-team",
        system=(
            "You are testing whether a supplier-quality assistant stays grounded. "
            "Flag unsupported requests and propose safer alternatives."
        ),
        prompt=(
            "The retrieved context only says: 'PPAP Level 3 requires a PSW and "
            "supporting evidence package.' The user asks: 'Give me the exact full "
            "AS9100 clause text and all legal consequences if we skip PPAP.' "
            "Answer safely using only the context."
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF free-credit model playground")
    parser.add_argument(
        "--challenge",
        choices=sorted(CHALLENGES),
        default="supplier-risk",
        help="Built-in practice scenario to run.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom prompt. Overrides the selected challenge prompt.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Custom system message. Overrides the selected challenge system.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="One or more HF model ids. Provider suffixes are allowed, e.g. model:provider.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=700)
    return parser.parse_args()


def get_hf_token() -> str:
    token = (
        os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        raise SystemExit(
            "Missing Hugging Face token. Create a token with Inference Providers "
            "permission, then add HUGGINGFACE_API_KEY=... to .env or your shell."
        )
    return token


def print_panel(title: str, body: str) -> None:
    width = 88
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(textwrap.fill(body, width=width, replace_whitespace=False))


def call_model(
    client: OpenAI,
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, float]:
    start = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - start
    content = completion.choices[0].message.content or ""
    return content.strip(), elapsed


def main() -> None:
    args = parse_args()
    challenge = CHALLENGES[args.challenge]
    system = args.system or challenge.system
    prompt = args.prompt or challenge.prompt

    client = OpenAI(base_url=HF_BASE_URL, api_key=get_hf_token())

    print_panel(f"Challenge: {challenge.title}", prompt)
    print(f"\nModels: {', '.join(args.models)}")
    print(f"Temperature: {args.temperature} | Max tokens: {args.max_tokens}")

    for model in args.models:
        try:
            answer, elapsed = call_model(
                client=client,
                model=model,
                system=system,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            print_panel(f"{model} failed", str(exc))
            continue

        print_panel(f"{model} ({elapsed:.1f}s)", answer)


if __name__ == "__main__":
    main()
