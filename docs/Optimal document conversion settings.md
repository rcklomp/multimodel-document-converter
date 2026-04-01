# Optimal setting for the conversion of the available documents

## Technical manuals

```bash
mmrag-v2 process data/technical_manual/integra_u_en.pdf \
  --output-dir output/integra_final_6 \
  --profile-override technical_manual \
  --vision-provider openai \
  --vision-model "numarkdown-8b-thinking-mlxs" \
  --vision-base-url "http://192.168.10.11:1234/v1" \
  --api-key "lm-studio"
  --vlm-timeout 600 \
  --no-refiner \
  --no-force-table-vlm
```

```bash
mmrag-v2 process "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" \
  --output-dir output/rag_guide_kimothi_v5 \
  --vision-provider openai \
  --vision-model "numarkdown-8b-thinking-mlxs" \
  --vision-base-url "http://192.168.10.11:1234/v1" \
  --api-key "lm-studio" \
  --vlm-timeout 600 \
  --batch-size 3 \
  --sensitivity 0.6 \
  --no-refiner \
  --no-force-table-vlm
```

```bash
/Users/ronald/miniforge3/envs/mmrag-v2/bin/mmrag-v2 process \
  "data/technical_manual/Bourne K. Unlocking Data with Generative AI and RAG 2024.pdf" \
  --output-dir output/bourne_unlocking_data_with_rag_v2 \
  --vision-provider openai \
  --vision-model "numarkdown-8b-thinking-mlxs" \
  --vision-base-url "http://192.168.10.11:1234/v1" \
  --api-key "lm-studio" \
  --vlm-timeout 600 \
  --batch-size 3 \
  --no-refiner \
  --no-force-table-vlm
```
