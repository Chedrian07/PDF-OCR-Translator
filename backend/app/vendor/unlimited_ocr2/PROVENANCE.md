# Unlimited-OCR-2 vendor provenance

Source: https://huggingface.co/baidu/Unlimited-OCR-2/tree/c14071b549f4fcab10909f3c4acc6a2edb366947
Revision: `c14071b549f4fcab10909f3c4acc6a2edb366947` (2026-10-30 retrieval).
Original Python file SHA-256 values are in `upstream-sha256.json`.
Model config, tokenizer and safetensors use this same immutable revision through the Hub cache.
The old model has a separate repository/cache key and is never a fallback.

Local changes in `modeling_unlimitedocr.py`:

- Device-aware boolean image embedding assignment (preserves the prior MPS broadcast-mask fix).
- Image tensors follow model dtype; CPU defaults to fp32 and Metal to bf16 when supported.
- Streamer and cancellation callbacks in the official DASH generation loop.
- Generation state restored in upstream `finally`; the application serializes model use.
- Remove output-driven eval/cropping functions; use `app.engine.postprocess`, which uses
  literal coordinate parsing, clamping, v2 BLOCK/FORK normalization and the existing artifact contract.
- Model-input views share conservative washed-paper contrast recovery through
  `app.engine.image_input`; original image pixels remain unchanged for figure crops.

All other upstream Python files are byte-for-byte copies. CPU and Metal use the upstream
pure torch sparse fallback; CUDA serving uses the separately pinned official SGLang wheel.
See `docs/UNLIMITED_OCR_2.md` and its validation record for actual platform evidence.
