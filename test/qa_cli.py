"""CLI for the laser-welding multimodal QA service.

Usage:
    python qa_cli.py "激光焊接为什么产生气孔？"
    python qa_cli.py "这个焊缝有什么缺陷？" --image weld.jpg
    python qa_cli.py "..." --backend cloud --model qwen-vl-max --api-key sk-xxx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import QA_API_BASE, QA_API_KEY, QA_BACKEND, QA_MODEL, QA_TIMEOUT
from qa_service import LaserWeldingQA


def main() -> None:
    parser = argparse.ArgumentParser(description="Laser welding multimodal QA")
    parser.add_argument("question", help="question text, e.g. 激光焊接为什么产生气孔？")
    parser.add_argument("--image", default=None, help="optional image path for visual questions")
    parser.add_argument("--backend", default=None, help="cloud or local (default: config.py)")
    parser.add_argument("--model", default=None, help="model name (default: config.py)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key (cloud); local can omit")
    args = parser.parse_args()

    qa = LaserWeldingQA(
        backend=args.backend or QA_BACKEND,
        model=args.model or QA_MODEL,
        api_base=args.base_url or QA_API_BASE,
        api_key=args.api_key or QA_API_KEY,
        timeout=QA_TIMEOUT,
    )
    print(f"[QA] backend={qa.backend}  model={qa.model}")
    print(f"[QA] base_url={qa.api_base}")
    print("-" * 60)
    try:
        answer = qa.ask(args.question, image_path=args.image)
    except Exception as e:
        print(f"[QA] 调用失败: {e}", file=sys.stderr)
        sys.exit(1)
    print(answer)


if __name__ == "__main__":
    main()
