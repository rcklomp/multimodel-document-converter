"""
mmrag-v2 process "data/raw/Combat Aircraft - August 2025 UK.pdf" \
    --batch-size 5 \
    --vision-provider openai \
    --vision-model "llama-joycaption-beta-one-hf-llava-mmproj" \
    --api-key "lm-studio" \
    --vlm-timeout 300 \
    --output-dir "output/Combat Aircraft - August 2025 UK_V18_1"
"""

import requests

try:
    r = requests.get("http://192.168.10.11:1234/v1/models", timeout=5)
    print(f"Succes! Status: {r.status_code}")
except Exception as e:
    print(f"Fout: {e}")
