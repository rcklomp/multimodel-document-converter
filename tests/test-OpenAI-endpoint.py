"""
mmrag-v2 process "data/raw/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf" \
    --batch-size 5 \
    --vision-provider openai \
    --vision-model "llama-joycaption-beta-one-hf-llava-mmproj" \
    --api-key "lm-studio" \
    --vlm-timeout 300 \
    --output-dir "output/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters_V19"

mmrag-v2 process "data/raw/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters.pdf" \
    --batch-size 5 \
    --vision-provider openai \
    --vision-model "llama-joycaption-beta-one-hf-llava-mmproj" \
    --vision-base-url "http://192.168.10.11:1234/v1" \
    --api-key "lm-studio" \
    --vlm-timeout 300 \
    --enable-refiner \
    --refiner-provider openai \
    --refiner-model "mistral-7b-instruct-v0.3-mixed-6-8-bit" \
    --refiner-base-url "http://192.168.10.11:1234/v1" \
    --output-dir "output/ATZ.Elektronik.-.Effizientere.Software.Entwicklung.GERMAN.RETAiL.eBOOk-PDFWriters_V23"

mmrag-v2 process "data/raw/AIOS LLM Agent Operating System.pdf" \
    --batch-size 5 \
    --ocr-mode layout-aware \
    --enable-doctr \
    --vision-provider openai \
    --vision-model "llama-joycaption-beta-one-hf-llava-mmproj" \
    --vision-base-url "http://192.168.10.11:1234/v1" \
    --vlm-timeout 300 \
    --enable-refiner \
    --refiner-provider openai \
    --refiner-model "mistral-7b-instruct-v0.3-mixed-6-8-bit" \
    --refiner-base-url "http://192.168.10.11:1234/v1" \
    --refiner-max-edit 1.0 \
    --api-key "lm-studio" \
    --output-dir "output/AIOS LLM Agent Operating System_V19" \
    --verbose

/Users/ronald/miniforge3/envs/mmrag-v2/bin/python -m mmrag_v2.cli process "data/academic_journal/<FILE>.pdf" \
  --batch-size 5 --ocr-mode auto --enable-doctr --auto-safe \
  --vision-provider openai --vision-model "llama-joycaption-beta-one-hf-llava-mmproj" \
  --vision-base-url "http://192.168.10.11:1234/v1" --vlm-timeout 180 \
  --enable-refiner --refiner-provider openai \
  --refiner-model "mistral-7b-instruct-v0.3-mixed-6-8-bit" \
  --refiner-base-url "http://192.168.10.11:1234/v1" --refiner-max-edit 1.0 \
  --api-key "lm-studio" \
  --output-dir "output/<FILE>_v2.4.1-plus" --verbose > logs/<FILE>.log 2>&1
"""

import requests

try:
    r = requests.get("http://192.168.10.11:1234/v1/models", timeout=5)
    print(f"Succes! Status: {r.status_code}")
except Exception as e:
    print(f"Fout: {e}")
