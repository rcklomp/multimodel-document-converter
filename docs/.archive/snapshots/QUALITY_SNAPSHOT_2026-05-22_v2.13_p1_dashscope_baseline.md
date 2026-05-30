# Quality Snapshot 2026-05-22 — SOAK (mmrag_v2_8__qwen3_dashscope)

> **Status:** synthetic-soak report.
> Source: `output/soak/v2.13_p1_dashscope_baseline/work.jsonl`.
> Judge: Dashscope `qwen-max`. Generator: `qwen-max`. Embedder: `text-embedding-v4` (provider=dashscope). Collection: `mmrag_v2_8__qwen3_dashscope`. Reranker: `omlx`.
> No QA threshold; this snapshot is informational.

## 1. Corpus summary

- Sampled chunks: **259** across 33 docs.
- Queries generated: **518**.
- Queries judged: **518/518** (100.0%).

## 2. Headline metrics

| Metric | Value |
|---|---:|
| Recall@1 (gold chunk_id is top-1) | 55.0% (285/518) |
| Recall@5 (gold chunk_id in top-5) | 72.6% (376/518) |
| Recall@5 (gold doc_id in top-5)   | 93.1% (482/518) |
| Relevance score                   | 74.1% (768/1036) |
| Format score                      | 89.2% (924/1036) |
| Faithfulness score                | 65.9% (683/1036) |

## 3. Per-document metrics

| Doc | Queries | R@1 | R@5 (chunk) | R@5 (doc) | Relevance | Format | Faith |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIOS_LLM_Agent_Operating_System | 16 | 81.2% | 81.2% | 87.5% | 84.4% | 96.9% | 75.0% |
| ATZ_Elektronik_German | 16 | 75.0% | 93.8% | 100.0% | 78.1% | 81.2% | 62.5% |
| A_comprehensive_review_on_hybrid_electri | 16 | 56.2% | 68.8% | 93.8% | 62.5% | 84.4% | 59.4% |
| Adedeji_GenAI_Google_Cloud | 16 | 50.0% | 62.5% | 87.5% | 65.6% | 93.8% | 59.4% |
| ArcGIS_Python_Cookbook | 16 | 37.5% | 62.5% | 81.2% | 71.9% | 93.8% | 68.8% |
| Ayeva_Python_Patterns | 16 | 68.8% | 81.2% | 87.5% | 75.0% | 93.8% | 65.6% |
| Bourne_RAG_2024 | 16 | 31.2% | 56.2% | 81.2% | 71.9% | 87.5% | 56.2% |
| CarOK_voorraadtelling | 16 | 18.8% | 31.2% | 100.0% | 40.6% | 59.4% | 21.9% |
| ChatGPT_Praktijk_handboek | 16 | 50.0% | 75.0% | 100.0% | 75.0% | 100.0% | 68.8% |
| Chaubal_PyTorch_Projects | 16 | 50.0% | 62.5% | 93.8% | 65.6% | 96.9% | 62.5% |
| Combat_Aircraft_August_2025 | 16 | 75.0% | 87.5% | 100.0% | 75.0% | 75.0% | 71.9% |
| Cronin_GenAI_Models | 16 | 43.8% | 81.2% | 87.5% | 78.1% | 93.8% | 71.9% |
| Devlin_LLM_Agents | 16 | 56.2% | 62.5% | 75.0% | 59.4% | 75.0% | 53.1% |
| Earthship_Vol1 | 16 | 93.8% | 93.8% | 93.8% | 84.4% | 68.8% | 81.2% |
| Firearms | 16 | 56.2% | 87.5% | 100.0% | 87.5% | 87.5% | 71.9% |
| Fluent_Python | 16 | 37.5% | 93.8% | 100.0% | 71.9% | 100.0% | 59.4% |
| Form_betwistingsformulier | 6 | 66.7% | 83.3% | 83.3% | 66.7% | 100.0% | 58.3% |
| Greenhouse_Design | 16 | 62.5% | 87.5% | 100.0% | 84.4% | 81.2% | 81.2% |
| Hao_ML_Platform | 16 | 43.8% | 68.8% | 87.5% | 81.2% | 100.0% | 68.8% |
| HarryPotter_and_the_Sorcerers_Stone | 16 | 37.5% | 68.8% | 100.0% | 68.8% | 90.6% | 62.5% |
| Hybrid_electric_vehicles | 16 | 93.8% | 93.8% | 100.0% | 75.0% | 87.5% | 62.5% |
| IRJET_Modeling_of_Solar_PV | 16 | 75.0% | 81.2% | 93.8% | 75.0% | 87.5% | 68.8% |
| Integra_manual | 16 | 50.0% | 56.2% | 100.0% | 62.5% | 84.4% | 59.4% |
| Jungjun_AI_Agent | 16 | 50.0% | 68.8% | 87.5% | 75.0% | 100.0% | 68.8% |
| KI_En_ChatGPT_Praktische_Gids | 16 | 25.0% | 37.5% | 100.0% | 68.8% | 100.0% | 46.9% |
| Kimothi_RAG_Guide | 16 | 56.2% | 56.2% | 93.8% | 84.4% | 93.8% | 75.0% |
| Nagasubramanian_Agentic_AI | 16 | 37.5% | 43.8% | 75.0% | 59.4% | 100.0% | 50.0% |
| PCWorld_July_2025 | 16 | 43.8% | 87.5% | 100.0% | 78.1% | 100.0% | 71.9% |
| Python_Cookbook | 16 | 56.2% | 75.0% | 93.8% | 81.2% | 75.0% | 81.2% |
| Python_Distilled | 16 | 81.2% | 87.5% | 93.8% | 90.6% | 93.8% | 78.1% |
| Raieli_AI_Agents | 16 | 50.0% | 68.8% | 87.5% | 78.1% | 100.0% | 75.0% |
| Recent_Trends_in_Transportation | 16 | 68.8% | 100.0% | 100.0% | 84.4% | 68.8% | 81.2% |
| Sekar_MCP_Standard | 16 | 43.8% | 56.2% | 100.0% | 81.2% | 100.0% | 71.9% |

## 4. Weakest 15 (query, top-1) pairs — v2.10.x defect candidates

- **S0155.Q2** total=0/6 (r=0, f=0, faith=0)
  - Query: 'How do reflective agents differ from reactive ones in their approach?'
  - Gold doc: `Devlin_LLM_Agents` (chunk `5b915c809145_284_text_071ce51f`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=278 score=0.664046
  - Judge rationale: The chunk is not relevant to the query, has severe formatting issues, and does not provide a correct or useful answer.
- **S0006.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: "What determines the color version of the code lock's backlight?"
  - Gold doc: `Integra_manual` (chunk `9f3ade9d82e0_046_text_d41ede54`)
  - Top-1: `integra_u_en.pdf` p=47 score=0.63652
  - Judge rationale: The retrieved chunk discusses the functions of different LED colors on a code lock but does not provide information about what determines the color version of the backlight as requested in the user query, and it is not self-contained for this specific question.
- **S0006.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: "How is the color of the backlight indicated in the code lock's designation?"
  - Gold doc: `Integra_manual` (chunk `9f3ade9d82e0_046_text_d41ede54`)
  - Top-1: `integra_u_en.pdf` p=47 score=0.677203
  - Judge rationale: The retrieved chunk discusses the functions of different LED colors on a code lock but does not answer how the color of the backlight is indicated in the code lock's designation, and it has minor formatting issues with odd whitespace and structure.
- **S0053.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How does the journal protect its content from unauthorized reproduction?'
  - Gold doc: `ATZ_Elektronik_German` (chunk `6fccda8bd625_006_text_762320b5`)
  - Top-1: `Firearms.pdf` p=4 score=0.424783
  - Judge rationale: The retrieved chunk does not provide any information about how the journal protects its content from unauthorized reproduction and is a description of a page layout rather than the content itself.
- **S0071.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models use the oliefilter with an ink.ex.BTW Titel of 2,30?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_007_text_7b2e5fc3`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=10 score=0.686266
  - Judge rationale: The retrieved chunk discusses interior filters for different car models, which is not relevant to the user's query about oliefilter with a specific BTW Titel, and it provides no correct or useful information in this context.
- **S0100.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models are compatible with the oliefilter having art_nr_merk 61551?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_008_text_6eb2d7fc`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=6 score=0.685765
  - Judge rationale: The retrieved chunk does not answer the query about car models compatible with art_nr_merk 61551 and contains minor format issues.
- **S0153.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models use the oliefilter with art_nr_merk 64605?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_009_text_4518215c`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=7 score=0.689548
  - Judge rationale: The retrieved chunk does not contain information about the car models that use the oliefilter with art_nr_merk 64605, and the format has minor issues with truncation and odd whitespace.
- **S0168.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What are the categories of automation techniques mentioned?'
  - Gold doc: `Bourne_RAG_2024` (chunk `c480fb4e3164_338_text_58d13ec2`)
  - Top-1: `Nagasubramanian D. Agentic AI for Engineers.Architecting Goal-Driven System 2026.pdf` p=54 score=0.450626
  - Judge rationale: The chunk discusses examples of automation but does not list categories of automation techniques as requested; the format is readable but slightly truncated, and it would mislead a user looking for specific categories.
- **S0182.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Wat is de prijs van het luchtfilter met merknummer 93183412?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_004_text_9bdca6fe_o2`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=7 score=0.495791
  - Judge rationale: The retrieved chunk does not contain information about the price or the specific product number (merknummer 93183412) asked for in the user query, and it has minor format issues with odd whitespace and structure.
- **S0182.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Hoeveel kost een oliefilter van Opel met artikelnummer 5650367?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_004_text_9bdca6fe_o2`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=3 score=0.678621
  - Judge rationale: The retrieved chunk does not contain information about the specific Opel oil filter with the article number 5650367, and the format has minor issues with truncation and odd whitespace, making it misleading for the user query.
- **S0215.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the range of the labels before adjustment?'
  - Gold doc: `Chaubal_PyTorch_Projects` (chunk `57dd24faf967_128_text_e24c7185`)
  - Top-1: `Firearms.pdf` p=201 score=0.0
  - Judge rationale: The retrieved chunk is not relevant to the user query and has minor formatting issues, making it misleading for answering the question about label ranges.
- **S0240.Q2** total=1/6 (r=1, f=0, faith=0)
  - Query: 'How do DCGANs fit into the trends of deep convolutional models?'
  - Gold doc: `Cronin_GenAI_Models` (chunk `0054f66093d6_634_text_79d7449b_s1_o2`)
  - Top-1: `Cronin I. Building and Training Generative AI Models. A Practical Guide...2026.pdf` p=634 score=0.549607
  - Judge rationale: The retrieved chunk is not well-formed and does not directly answer the query about DCGANs trends, instead it points to another section for more information.
- **S0249.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What does a biomedical agent use?'
  - Gold doc: `Devlin_LLM_Agents` (chunk `5b915c809145_261_text_5cc7d204`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=261 score=0.484337
  - Judge rationale: The chunk does not provide any information about what a biomedical agent uses, and it contains repeated words and phrases making it hard to read, with no self-contained or correct answer available.
- **S0001.Q1** total=2/6 (r=0, f=2, faith=0)
  - Query: 'What does the _parse_response method do with API responses?'
  - Gold doc: `Jungjun_AI_Agent` (chunk `6afeb55a9449_119_text_14cd977d`)
  - Top-1: `Programming ArcGIS with Python Cookbook.pdf` p=289 score=0.518866
  - Judge rationale: The chunk discusses making and parsing REST requests in Python, which is unrelated to the _parse_response method for handling API responses and tool calls.
- **S0005.Q2** total=2/6 (r=0, f=2, faith=0)
  - Query: 'How did Ron try to convince Neville to let them go?'
  - Gold doc: `HarryPotter_and_the_Sorcerers_Stone` (chunk `f0a0beca0506_284_text_17ff0064`)
  - Top-1: `HarryPotter_and_the_Sorcerers_Stone.pdf` p=168 score=0.627323
  - Judge rationale: The retrieved chunk does not address how Ron tried to convince Neville to let them go, instead it discusses a different scenario with Neville and the group.

## 5. Methodology

- Sampled 259 text chunks (≥ 150 chars, ≤ 40% code-like lines, no advertisement keywords). Stratified across the 34-doc canonical corpus.
- Each chunk → 2 queries generated by `qwen-max` (temperature 0.3).
- Each query → top-5 retrieved from `mmrag_v2_8__qwen3_dashscope` via `dashscope` provider, model `text-embedding-v4`.
- Each top-1 chunk → graded by `qwen-max` (temperature 0.0) on relevance / format / faithfulness, each 0-2.
- Gold passage is shown to the judge for context; the judge is instructed NOT to penalize a different-chunk same-document retrieval.

## 6. Revision log

| Date | Change |
|---|---|
| 2026-05-22 | Initial v2.10.0-rc1 soak snapshot. |