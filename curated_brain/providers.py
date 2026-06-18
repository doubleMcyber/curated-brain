"""Real frozen-model providers (PRD §5.3) behind the ``Embedder`` / ``LLM`` protocols.

These are the production counterparts to the deterministic fakes in :mod:`fakes`. The
fakes stay the byte-deterministic *test doubles* (AC-1); these run real local models so
the pipeline is no longer faked end-to-end. Same protocols, so nothing in the core path
changes — only which object is injected.

Design constraints:

* **Lazy + soft-dependency.** Importing this module never pulls in ``torch`` /
  ``sentence_transformers`` / ``transformers``; the heavy stack is imported on first use,
  and its absence raises an actionable error pointing at the ``local`` extra. This keeps
  the offline gate (which runs on the fakes) free of the model stack.
* **Unit-norm vectors.** Embeddings are L2-normalized to match the ``Embedder`` contract,
  so cosine == dot product exactly as with the fake embedder.
* **Greedy decoding.** The local LLM decodes deterministically (no sampling), the closest
  a real model gets to the fakes' byte-determinism; pair with :mod:`cassette` for fully
  reproducible CI runs.
"""

from __future__ import annotations

import numpy as np

# Best-known output dimensions, so ``.dim`` is available without loading the model (which
# would otherwise be triggered by a plain ``hasattr``/``isinstance`` Protocol check). The
# real value overwrites this once the model loads.
_KNOWN_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/e5-small-v2": 384,
    "intfloat/e5-base-v2": 768,
}


class SentenceTransformerEmbedder:
    """Real neural embedder (default ``BAAI/bge-small-en-v1.5``) via sentence-transformers.

    Loads lazily; the model name is recorded into ``model_id`` so vectors can be
    re-embedded when the model is upgraded (PRD §12). Conforms to the ``Embedder``
    protocol, so it drops straight into :class:`~curated_brain.backend.CuratedBrain`.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", *,
                 device: str | None = None) -> None:
        self.model_name = model_name
        self.model_id = f"st:{model_name}"
        self.dim = _KNOWN_DIMS.get(model_name, 0)
        self._device = device
        self._model = None  # lazily constructed real model

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "SentenceTransformerEmbedder requires the 'local' extra: "
                    "pip install 'curated-brain[local]'"
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self._device)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, text: str) -> np.ndarray:
        m = self._ensure()
        v = m.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float64)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        m = self._ensure()
        v = m.encode(list(texts), normalize_embeddings=True)
        return np.asarray(v, dtype=np.float64)


class TransformersLLM:
    """Real local chat model via 🤗 ``transformers`` (e.g. a cached Qwen / Mistral instruct
    model). Greedy decoding for determinism; used for the *optional* LLM signals
    (summarization, claim extraction, contradiction adjudication) — never the core path.

    Loaded lazily and kept off the default gate; pair with the cassette layer for CI.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", *,
                 device: str | None = None, max_new_tokens: int = 256) -> None:
        self.model_name = model_name
        self.model_id = f"hf:{model_name}"
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._tok = None
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                import torch  # noqa: F401
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "TransformersLLM requires the 'local' extra: "
                    "pip install 'curated-brain[local]'"
                ) from e
            import torch
            device = self._device or ("mps" if torch.backends.mps.is_available()
                                      else "cuda" if torch.cuda.is_available() else "cpu")
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype="auto").to(device)
            self._dev = device
        return self._model

    def complete(self, prompt: str) -> str:
        import torch
        model = self._ensure()
        msgs = [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._dev)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                 do_sample=False, num_beams=1,
                                 pad_token_id=self._tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()


# ----------------------------------------------------------------- remote providers --
# OpenAI-compatible HTTP providers. The point for Track D: Curated Brain and every rival
# (Mem0/Letta/Zep) can be pointed at the SAME endpoint + model, so the head-to-head varies
# only the memory layer — and it dodges the local-CPU bottleneck (use a hosted or vLLM
# endpoint). Stdlib-only HTTP (no new dependency); a `post` seam keeps them offline-testable.

def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize so cosine == dot, matching the Embedder contract (0-vector stays 0)."""
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _http_post_json(url: str, body: dict, api_key: str | None, timeout: float) -> dict:
    import json
    import urllib.request
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted base_url)
        return json.loads(resp.read().decode("utf-8"))


class OpenAICompatEmbedder:
    """Embedder backed by any OpenAI-compatible ``/embeddings`` endpoint (OpenAI, vLLM,
    Ollama's OpenAI mode, Together, …). ``dim`` is required so the vector tier can be sized
    without a network call. Pass ``post`` (a ``(path, body) -> dict`` callable) to inject a
    transport in tests; production uses stdlib HTTP."""

    def __init__(self, model: str, *, dim: int,
                 base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, timeout: float = 30.0, post=None) -> None:
        self.model = model
        self.model_id = f"openai:{model}"
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._post = post

    def _request(self, path: str, body: dict) -> dict:
        if self._post is not None:
            return self._post(path, body)
        return _http_post_json(self.base_url + path, body, self._api_key, self._timeout)

    def embed(self, text: str) -> np.ndarray:
        data = self._request("/embeddings", {"model": self.model, "input": text})
        return _unit(np.asarray(data["data"][0]["embedding"], dtype=np.float64))

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        data = self._request("/embeddings", {"model": self.model, "input": list(texts)})
        rows = sorted(data["data"], key=lambda d: d["index"])  # preserve input order
        return np.vstack([_unit(np.asarray(r["embedding"], dtype=np.float64)) for r in rows])


class OpenAICompatLLM:
    """Chat LLM backed by any OpenAI-compatible ``/chat/completions`` endpoint. Temperature
    defaults to 0 (greedy) for reproducibility. Pass ``post`` to inject a transport in tests."""

    def __init__(self, model: str, *, base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, max_tokens: int = 256,
                 temperature: float = 0.0, timeout: float = 60.0, post=None) -> None:
        self.model = model
        self.model_id = f"openai:{model}"
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._timeout = timeout
        self._post = post

    def _request(self, path: str, body: dict) -> dict:
        if self._post is not None:
            return self._post(path, body)
        return _http_post_json(self.base_url + path, body, self._api_key, self._timeout)

    def complete(self, prompt: str) -> str:
        body = {"model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens, "temperature": self.temperature}
        data = self._request("/chat/completions", body)
        return data["choices"][0]["message"]["content"].strip()
