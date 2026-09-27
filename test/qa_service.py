"""
robot_monitor v0.5 - Laser welding multimodal QA service

Unified OpenAI-compatible client that works with BOTH:
  - cloud models: Aliyun Bailian (qwen-vl), SiliconFlow, OpenAI GPT-4o, ...
  - local models: Ollama / LM Studio / vLLM (OpenAI-compatible endpoint)

Switch between them in test/config.py:
    QA_BACKEND = "cloud"  -> this computer calls a cloud API (needs a key)
    QA_BACKEND = "local"  -> studio computer runs the model locally (Ollama)
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

SYSTEM_PROMPT = (
    "你是激光焊接领域的资深工艺与质量专家。请用中文回答，结合焊接工艺、材料、"
    "设备与维护知识，给出准确、可操作的解答。涉及缺陷时说明成因、危害与改善措施；"
    "涉及工艺参数时给出合理范围并解释调整方向；涉及设备维护时给出检查项和建议。"
    "如果提供了图像，先描述图像内容，再结合图像回答用户问题。不确定时明确说明，"
    "不要编造。"
)


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/png")


class LaserWeldingQA:
    """OpenAI-compatible multimodal QA client (cloud or local)."""

    def __init__(
        self,
        backend: str = "local",
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.model = model or ("qwen3-vl:8b" if backend == "local" else "qwen-vl-max")
        self.api_base = api_base or (
            "http://localhost:11434/v1"
            if backend == "local"
            else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.api_key = api_key or ("local" if backend == "local" else "")
        self.timeout = timeout
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package is not installed; run: pip install openai"
            ) from e
        if self.backend == "cloud" and not self.api_key:
            raise RuntimeError(
                "云端问答需要 API Key：请在环境变量 QA_API_KEY 中设置，"
                "或直接修改 test/config.py 的 QA_API_KEY"
            )
        self._client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        return self._client

    def ask(
        self,
        question: str,
        image_path: Optional[str | Path] = None,
        image_base64: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Ask a question, optionally with an image (path or base64 data URL)."""
        client = self._ensure_client()

        user_content: list[dict] = []
        if image_base64:
            data = image_base64
            if data.startswith("data:") and "," in data:
                data = data.split(",", 1)[1]
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{data}"},
            })
        elif image_path:
            p = Path(image_path)
            if not p.exists():
                raise FileNotFoundError(f"image not found: {p}")
            b64 = base64.b64encode(p.read_bytes()).decode()
            mime = _guess_mime(p)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        user_content.append({"type": "text", "text": question})
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    def health(self) -> str:
        """Quick connectivity check (may raise on failure)."""
        client = self._ensure_client()
        models = client.models.list()
        names = [m.id for m in models.data][:5]
        return f"backend={self.backend} model={self.model} api_base={self.api_base} ok, sample models={names}"
