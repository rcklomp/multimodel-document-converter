"""
mmrag-v2 process "data/raw/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf" \
    --batch-size 5 \
    --vision-provider openai \
    --vision-model "llama-joycaption-beta-one-hf-llava-mmproj" \
    --api-key "lm-studio" \
    --vlm-timeout 300 \
    --output-dir "output/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters_V18_2_2"
"""

import requests

try:
    r = requests.get("http://192.168.10.11:1234/v1/models", timeout=5)
    print(f"Succes! Status: {r.status_code}")
except Exception as e:
    print(f"Fout: {e}")
