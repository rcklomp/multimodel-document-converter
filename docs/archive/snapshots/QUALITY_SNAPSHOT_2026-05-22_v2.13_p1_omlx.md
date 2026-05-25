# Quality Snapshot 2026-05-22 — SOAK (mmrag_v2_8__qwen3_local)

> **Status:** synthetic-soak report.
> Source: `output/soak/v2.13_p1_omlx/work.jsonl`.
> Judge: Dashscope `qwen-max`. Generator: `qwen-max`. Embedder: `Qwen3-Embedding-8B-mxfp8` (provider=omlx). Collection: `mmrag_v2_8__qwen3_local`. Reranker: `omlx`.
> No QA threshold; this snapshot is informational.

## 1. Corpus summary

- Sampled chunks: **259** across 33 docs.
- Queries generated: **518**.
- Queries judged: **518/518** (100.0%).

## 2. Headline metrics

| Metric | Value |
|---|---:|
| Recall@1 (gold chunk_id is top-1) | 57.5% (298/518) |
| Recall@5 (gold chunk_id in top-5) | 78.0% (404/518) |
| Recall@5 (gold doc_id in top-5)   | 95.2% (493/518) |
| Relevance score                   | 74.6% (773/1036) |
| Format score                      | 92.9% (962/1036) |
| Faithfulness score                | 66.9% (693/1036) |

## 3. Per-document metrics

| Doc | Queries | R@1 | R@5 (chunk) | R@5 (doc) | Relevance | Format | Faith |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIOS_LLM_Agent_Operating_System | 16 | 75.0% | 81.2% | 87.5% | 75.0% | 96.9% | 62.5% |
| ATZ_Elektronik_German | 16 | 62.5% | 68.8% | 87.5% | 59.4% | 96.9% | 50.0% |
| A_comprehensive_review_on_hybrid_electri | 16 | 50.0% | 68.8% | 93.8% | 59.4% | 96.9% | 50.0% |
| Adedeji_GenAI_Google_Cloud | 16 | 68.8% | 75.0% | 93.8% | 68.8% | 96.9% | 65.6% |
| ArcGIS_Python_Cookbook | 16 | 43.8% | 81.2% | 87.5% | 75.0% | 96.9% | 68.8% |
| Ayeva_Python_Patterns | 16 | 68.8% | 87.5% | 93.8% | 84.4% | 96.9% | 75.0% |
| Bourne_RAG_2024 | 16 | 31.2% | 50.0% | 100.0% | 71.9% | 96.9% | 62.5% |
| CarOK_voorraadtelling | 16 | 12.5% | 43.8% | 100.0% | 37.5% | 71.9% | 25.0% |
| ChatGPT_Praktijk_handboek | 16 | 75.0% | 93.8% | 100.0% | 84.4% | 100.0% | 81.2% |
| Chaubal_PyTorch_Projects | 16 | 56.2% | 75.0% | 100.0% | 84.4% | 96.9% | 84.4% |
| Combat_Aircraft_August_2025 | 16 | 68.8% | 87.5% | 93.8% | 71.9% | 90.6% | 68.8% |
| Cronin_GenAI_Models | 16 | 81.2% | 93.8% | 100.0% | 81.2% | 96.9% | 78.1% |
| Devlin_LLM_Agents | 16 | 75.0% | 75.0% | 87.5% | 65.6% | 75.0% | 56.2% |
| Earthship_Vol1 | 16 | 93.8% | 93.8% | 93.8% | 81.2% | 62.5% | 78.1% |
| Firearms | 16 | 56.2% | 93.8% | 100.0% | 90.6% | 84.4% | 75.0% |
| Fluent_Python | 16 | 37.5% | 93.8% | 100.0% | 78.1% | 96.9% | 68.8% |
| Form_betwistingsformulier | 6 | 66.7% | 83.3% | 83.3% | 66.7% | 100.0% | 66.7% |
| Greenhouse_Design | 16 | 50.0% | 68.8% | 100.0% | 68.8% | 87.5% | 59.4% |
| Hao_ML_Platform | 16 | 50.0% | 75.0% | 100.0% | 84.4% | 100.0% | 71.9% |
| HarryPotter_and_the_Sorcerers_Stone | 16 | 37.5% | 87.5% | 100.0% | 75.0% | 100.0% | 65.6% |
| Hybrid_electric_vehicles | 16 | 81.2% | 81.2% | 100.0% | 65.6% | 93.8% | 59.4% |
| IRJET_Modeling_of_Solar_PV | 16 | 62.5% | 75.0% | 93.8% | 75.0% | 87.5% | 65.6% |
| Integra_manual | 16 | 62.5% | 87.5% | 100.0% | 81.2% | 100.0% | 75.0% |
| Jungjun_AI_Agent | 16 | 56.2% | 68.8% | 81.2% | 71.9% | 100.0% | 71.9% |
| KI_En_ChatGPT_Praktische_Gids | 16 | 31.2% | 37.5% | 100.0% | 65.6% | 100.0% | 46.9% |
| Kimothi_RAG_Guide | 16 | 75.0% | 87.5% | 93.8% | 90.6% | 100.0% | 90.6% |
| Nagasubramanian_Agentic_AI | 16 | 50.0% | 68.8% | 93.8% | 75.0% | 100.0% | 68.8% |
| PCWorld_July_2025 | 16 | 37.5% | 93.8% | 100.0% | 81.2% | 96.9% | 75.0% |
| Python_Cookbook | 16 | 43.8% | 93.8% | 93.8% | 78.1% | 75.0% | 71.9% |
| Python_Distilled | 16 | 75.0% | 75.0% | 93.8% | 75.0% | 87.5% | 62.5% |
| Raieli_AI_Agents | 16 | 62.5% | 81.2% | 87.5% | 84.4% | 100.0% | 75.0% |
| Recent_Trends_in_Transportation | 16 | 62.5% | 100.0% | 100.0% | 84.4% | 87.5% | 78.1% |
| Sekar_MCP_Standard | 16 | 43.8% | 50.0% | 93.8% | 65.6% | 100.0% | 53.1% |

## 4. Weakest 15 (query, top-1) pairs — v2.10.x defect candidates

- **S0179.Q2** total=0/6 (r=0, f=0, faith=0)
  - Query: 'How does the author describe the role of code in projects?'
  - Gold doc: `Python_Cookbook` (chunk `0326dff0bbb4_476_text_b828c800`)
  - Top-1: `Adedeji A. GenAI on Google Cloud. Enterprise Generative AI Systems...Agents 2026.pdf` p=246 score=0.563675
  - Judge rationale: The retrieved chunk is not related to the user query and contains broken, non-prose content, making it irrelevant and misleading for the user's question.
- **S0024.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Why is it important to monitor PAR for plants?'
  - Gold doc: `Greenhouse_Design` (chunk `8b79e9dca3ae_220_text_5892aff4`)
  - Top-1: `Greenhouse Design and Control by Pedro Ponce.pdf` p=15 score=0.566563
  - Judge rationale: The retrieved chunk does not answer the user query about monitoring PAR for plants and is a list of figure references, making it irrelevant and not self-contained for the question asked.
- **S0074.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How is the PWM signal generated in the proposed model?'
  - Gold doc: `IRJET_Modeling_of_Solar_PV` (chunk `b670f5359e9c_006_text_bc690797`)
  - Top-1: `Greenhouse Design and Control by Pedro Ponce.pdf` p=175 score=0.503628
  - Judge rationale: The retrieved chunk discusses a different topic (fuzzy-PD controller for temperature and light intensity) and does not answer the query about PWM signal generation; it also contains some minor formatting issues and would mislead the user on the specific question asked.
- **S0102.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the recommended use of the amperometric method and why?'
  - Gold doc: `Greenhouse_Design` (chunk `8b79e9dca3ae_223_text_f1f71761`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=79 score=0.0
  - Judge rationale: The retrieved chunk is about building materials for Earthships and does not address the amperometric method or its recommended use, and it has minor formatting issues with spacing and punctuation.
- **S0107.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How can bridging internal silos benefit an organization?'
  - Gold doc: `Devlin_LLM_Agents` (chunk `5b915c809145_185_text_f8cf71aa`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=185 score=0.467257
  - Judge rationale: The chunk does not answer the query about benefits of bridging internal silos and has repeated words making it hard to read and understand.
- **S0111.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How does the dual-current loop control algorithm enhance HEV performance?'
  - Gold doc: `A_comprehensive_review_on_hybrid_electri` (chunk `1b6ba953d1f4_015_text_10bccbac`)
  - Top-1: `Hybrid_electric_vehicles_and_their_challenges.pdf` p=10 score=0.649575
  - Judge rationale: The retrieved chunk does not address the dual-current loop control algorithm or its impact on HEV performance and is only marginally related to the topic of HEVs, with some minor formatting issues.
- **S0153.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models use the oliefilter with art_nr_merk 64605?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_009_text_4518215c`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=6 score=0.741406
  - Judge rationale: The chunk does not mention the specific art_nr_merk 64605 or any related car models, and it has minor formatting issues with odd spacing and truncation.
- **S0167.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What argument is not needed in this example compared to the spin function?'
  - Gold doc: `Fluent_Python` (chunk `1e7e436164a3_568_text_48d62253`)
  - Top-1: `Fluent Python Luciano Ramalho 2015.pdf` p=566 score=0.543729
  - Judge rationale: The retrieved chunk does not address the user's query about the argument not needed and contains truncated code, making it neither relevant nor self-contained for a correct answer.
- **S0182.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Hoeveel kost een oliefilter van Opel met artikelnummer 5650367?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_004_text_9bdca6fe_o2`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=3 score=0.663639
  - Judge rationale: The chunk does not contain information about the specific Opel oil filter (5650367) requested and is poorly formatted with repeated and irrelevant information, making it neither relevant nor faithful to the query.
- **S0200.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How much does the oil filter for the Ford Escort cost excluding VAT?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_007_text_672f1c0b`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=7 score=0.6377
  - Judge rationale: The chunk does not contain information about the oil filter for the Ford Escort and its cost, and it has minor formatting issues with odd whitespace.
- **S0204.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the value of Egasoline in kWh per gallon?'
  - Gold doc: `Hybrid_electric_vehicles` (chunk `2baf312fdd78_010_text_518fc70a`)
  - Top-1: `A_comprehensive_review_on_hybrid_electri.pdf` p=11 score=0.446001
  - Judge rationale: The retrieved chunk does not contain information about the value of Egasoline in kWh per gallon and is a list of costs, which is not relevant to the query; the format is a bit odd with excessive use of periods but still readable; the faithfulness score is 0 as the information provided is not related
- **S0238.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the price of the General Motors 93165213 10W-40 1L motor oil?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_004_text_9bdca6fe`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=4 score=0.798068
  - Judge rationale: The retrieved chunk does not contain information about the price of the specific motor oil requested, and the table is slightly truncated but still readable.
- **S0240.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How do DCGANs fit into the trends of deep convolutional models?'
  - Gold doc: `Cronin_GenAI_Models` (chunk `0054f66093d6_634_text_79d7449b_s1_o2`)
  - Top-1: `Cronin I. Building and Training Generative AI Models. A Practical Guide...2026.pdf` p=634 score=0.610945
  - Judge rationale: The chunk does not provide any information about how DCGANs fit into the trends of deep convolutional models and is mostly a list of unrelated topics with some formatting issues.
- **S0249.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What does a biomedical agent use?'
  - Gold doc: `Devlin_LLM_Agents` (chunk `5b915c809145_261_text_5cc7d204`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=261 score=0.576096
  - Judge rationale: The chunk does not provide any information on what a biomedical agent uses and has minor formatting issues with repetitive words and odd spacing.
- **S0017.Q1** total=2/6 (r=0, f=2, faith=0)
  - Query: 'What is the default value of output_path.path set to?'
  - Gold doc: `Hao_ML_Platform` (chunk `70930ff6f3a8_248_text_8ca5e581`)
  - Top-1: `Hao B. Machine Learning Platform Engineering. Build...for ML and AI systems 2026.pdf` p=222 score=0.500222
  - Judge rationale: The retrieved chunk does not answer the query about the default value of output_path.path and instead provides an example of using Output in a KFP component, which is not relevant to the question asked.

## 5. Methodology

- Sampled 259 text chunks (≥ 150 chars, ≤ 40% code-like lines, no advertisement keywords). Stratified across the 34-doc canonical corpus.
- Each chunk → 2 queries generated by `qwen-max` (temperature 0.3).
- Each query → top-5 retrieved from `mmrag_v2_8__qwen3_local` via `omlx` provider, model `Qwen3-Embedding-8B-mxfp8`.
- Each top-1 chunk → graded by `qwen-max` (temperature 0.0) on relevance / format / faithfulness, each 0-2.
- Gold passage is shown to the judge for context; the judge is instructed NOT to penalize a different-chunk same-document retrieval.

## 6. Revision log

| Date | Change |
|---|---|
| 2026-05-22 | Initial v2.10.0-rc1 soak snapshot. |