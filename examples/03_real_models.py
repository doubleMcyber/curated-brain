"""Real local models: raw text in, facts extracted automatically — no spoon-feeding.

Requires the optional model stack and a small cached chat model:

    pip install -e '.[local]'
    python examples/03_real_models.py

A local chat model runs on CPU here (some models mis-handle Apple MPS). Swap in any
instruct model via the model name. This is illustrative and not part of the offline gate.
"""

from curated_brain import (
    CuratedBrain,
    LLMExtractor,
    SentenceTransformerEmbedder,
    TransformersLLM,
)


def main() -> None:
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    llm = TransformersLLM("Qwen/Qwen2.5-1.5B-Instruct", device="cpu")
    cb = CuratedBrain(embedder=embedder, dim=embedder.dim, extractor=LLMExtractor(llm))

    # No metadata.fact — the structured tier is populated from the raw text by the extractor.
    cb.write("Erin moved to Vienna last spring.", session_id="s1", timestamp=0.0)
    cb.write("Bob was promoted to engineering manager.", session_id="s1", timestamp=1.0)

    print("Extracted from raw text:")
    print("  Erin's city :", cb.answer_structured("Erin", "city"))
    print("  Bob's role  :", cb.answer_structured("Bob", "role"))


if __name__ == "__main__":
    main()
