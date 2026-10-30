"""Resolve a pinned snapshot, then exec the official Unlimited-OCR-2 SGLang wheel."""

import json
import os
import sys

from huggingface_hub import snapshot_download

MODEL_ID = "baidu/Unlimited-OCR-2"
REVISION = "c14071b549f4fcab10909f3c4acc6a2edb366947"


def command():
    snapshot = snapshot_download(
        MODEL_ID, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "*.py"]
    )
    for key, value in {
        "SGLANG_UOCR_EANABLE_DOC_LIMIT": "1",
        "SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW": "10",
        "SGLANG_UOCR_DOC_BASE_PAGE_WINDOW": "10",
        "SGLANG_UOCR_MAX_PAGE_TOKENS": "20000",
        "SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB": "64",
        "SGLANG_UOCR_SHM_WRAP_MODEL_DATA": "1",
        "SGLANG_UOCR_SHM_WRAP_MIN_MB": "1",
        "SGLANG_VLM_CACHE_SIZE_MB": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }.items():
        os.environ.setdefault(key, value)
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        snapshot,
        "--served-model-name",
        "Unlimited-OCR-2",
        "--chat-template",
        "unlimited-ocr",
        "--tokenizer-backend",
        "fastokens",
        "--tokenizer-worker-num",
        "1",
        "--enable-custom-logit-processor",
        "--enable-streaming-session",
        "--allow-auto-truncate",
        "--disable-radix-cache",
        "--mm-preprocess-cache-size-mb",
        "0",
        "--mem-fraction-static",
        os.environ.get("SGLANG_MEM_FRACTION", "0.65"),
        "--max-running-requests",
        os.environ.get("SGLANG_CONCURRENCY", "1"),
        "--max-prefill-tokens",
        "2048",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        os.environ.get("SGLANG_CONCURRENCY", "1"),
        "--cuda-graph-backend-prefill",
        "full",
        "--cuda-graph-max-bs-prefill",
        "2048",
        "--cuda-graph-bs-prefill",
        "1024",
        "2048",
        "--cuda-graph-config",
        json.dumps(
            {
                "prefill": {
                    "backend": "full",
                    "full_prefill_max_req": int(
                        os.environ.get("SGLANG_CONCURRENCY", "1")
                    ),
                }
            }
        ),
        "--host",
        os.environ.get("SGLANG_HOST", "127.0.0.1"),
        "--port",
        "10000",
        *sys.argv[1:],
    ]


if __name__ == "__main__":
    args = command()
    os.execv(args[0], args)
