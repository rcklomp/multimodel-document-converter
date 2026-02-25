# Optimal setting for the conversion of the available documents

## Technical manuals

```bash
mmrag-v2 process data/technical_manual/integra_u_en.pdf \
  --output-dir output/integra_final_5 \
  --profile-override technical_manual \
  --vision-provider openai \
  --vision-model "numarkdown-8b-thinking-mlxs" \
  --vision-base-url "http://192.168.10.11:1234/v1" \
  --api-key "lm-studio"
  --vlm-timeout 600 \
  --no-refiner \
  --no-force-table-vlm  \
```

```bash
mmrag-v2 process "data/technical_manual/A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf" \
  --output-dir output/rag_guide_kimothi_v1 \
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