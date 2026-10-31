"""Resolve a pinned snapshot, then exec the official Unlimited-OCR-2 SGLang wheel."""

import argparse
import json
import os
import sys
import tempfile

from huggingface_hub import snapshot_download

MODEL_ID = "baidu/Unlimited-OCR-2"
REVISION = "c14071b549f4fcab10909f3c4acc6a2edb366947"


def command(argv=None):
    # This is an OCR-only service. Do not forward arbitrary SGLang arguments:
    # config files/aliases must not re-enable optional grammar/cache code paths.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--host", default=os.environ.get("SGLANG_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=10000)
    args = parser.parse_args(argv)
    if not 0 < args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    # Independent of the disabled grammar path, never reuse a cache another
    # principal could have populated. mkdtemp creates an owner-only 0700 directory.
    os.environ["OUTLINES_CACHE_DIR"] = tempfile.mkdtemp(prefix="uocr2-outlines-")
    os.environ["SGLANG_DISABLE_OUTLINES_DISK_CACHE"] = "true"
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
        "--grammar-backend",
        "none",
        "--disable-outlines-disk-cache",
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
        args.host,
        "--port",
        str(args.port),
    ]


if __name__ == "__main__":
    args = command()
    os.execv(args[0], args)
