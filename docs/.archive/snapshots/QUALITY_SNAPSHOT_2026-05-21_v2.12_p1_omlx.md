# Quality Snapshot 2026-05-21 — SOAK (mmrag_v2_8__qwen3_dashscope)

> **Status:** synthetic-soak report.
> Source: `output/soak/v2.12_p1_omlx/work.jsonl`.
> Judge: Dashscope `qwen-max`. Generator: `qwen-max`. Embedder: `text-embedding-v4` (provider=dashscope). Collection: `mmrag_v2_8__qwen3_dashscope`. Reranker: `omlx`.
> No QA threshold; this snapshot is informational.

## 1. Corpus summary

- Sampled chunks: **259** across 33 docs.
- Queries generated: **518**.
- Queries judged: **518/518** (100.0%).

## 2. Headline metrics

| Metric | Value |
|---|---:|
| Recall@1 (gold chunk_id is top-1) | 61.8% (320/518) |
| Recall@5 (gold chunk_id in top-5) | 81.3% (421/518) |
| Recall@5 (gold doc_id in top-5)   | 95.2% (493/518) |
| Relevance score                   | 78.3% (811/1036) |
| Format score                      | 89.0% (922/1036) |
| Faithfulness score                | 69.4% (719/1036) |

## 3. Per-document metrics

| Doc | Queries | R@1 | R@5 (chunk) | R@5 (doc) | Relevance | Format | Faith |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIOS_LLM_Agent_Operating_System | 16 | 81.2% | 81.2% | 87.5% | 84.4% | 96.9% | 71.9% |
| ATZ_Elektronik_German | 16 | 75.0% | 87.5% | 93.8% | 65.6% | 84.4% | 50.0% |
| A_comprehensive_review_on_hybrid_electri | 16 | 75.0% | 93.8% | 93.8% | 68.8% | 81.2% | 59.4% |
| Adedeji_GenAI_Google_Cloud | 16 | 68.8% | 100.0% | 100.0% | 84.4% | 90.6% | 78.1% |
| ArcGIS_Python_Cookbook | 16 | 43.8% | 68.8% | 87.5% | 78.1% | 93.8% | 71.9% |
| Ayeva_Python_Patterns | 16 | 81.2% | 93.8% | 100.0% | 90.6% | 87.5% | 71.9% |
| Bourne_RAG_2024 | 16 | 50.0% | 75.0% | 93.8% | 78.1% | 90.6% | 65.6% |
| CarOK_voorraadtelling | 16 | 25.0% | 50.0% | 100.0% | 43.8% | 75.0% | 28.1% |
| ChatGPT_Praktijk_handboek | 16 | 56.2% | 75.0% | 87.5% | 78.1% | 100.0% | 71.9% |
| Chaubal_PyTorch_Projects | 16 | 68.8% | 87.5% | 100.0% | 81.2% | 100.0% | 78.1% |
| Combat_Aircraft_August_2025 | 16 | 81.2% | 93.8% | 100.0% | 81.2% | 75.0% | 78.1% |
| Cronin_GenAI_Models | 16 | 56.2% | 68.8% | 81.2% | 75.0% | 96.9% | 71.9% |
| Devlin_LLM_Agents | 16 | 75.0% | 75.0% | 81.2% | 68.8% | 71.9% | 62.5% |
| Earthship_Vol1 | 16 | 87.5% | 93.8% | 100.0% | 71.9% | 62.5% | 59.4% |
| Firearms | 16 | 62.5% | 87.5% | 100.0% | 84.4% | 65.6% | 75.0% |
| Fluent_Python | 16 | 43.8% | 87.5% | 100.0% | 75.0% | 100.0% | 62.5% |
| Form_betwistingsformulier | 6 | 50.0% | 83.3% | 83.3% | 58.3% | 100.0% | 50.0% |
| Greenhouse_Design | 16 | 56.2% | 100.0% | 100.0% | 87.5% | 84.4% | 78.1% |
| Hao_ML_Platform | 16 | 50.0% | 68.8% | 93.8% | 78.1% | 100.0% | 68.8% |
| HarryPotter_and_the_Sorcerers_Stone | 16 | 56.2% | 87.5% | 100.0% | 87.5% | 90.6% | 71.9% |
| Hybrid_electric_vehicles | 16 | 87.5% | 93.8% | 100.0% | 81.2% | 84.4% | 75.0% |
| IRJET_Modeling_of_Solar_PV | 16 | 75.0% | 81.2% | 100.0% | 87.5% | 87.5% | 75.0% |
| Integra_manual | 16 | 50.0% | 75.0% | 100.0% | 71.9% | 90.6% | 68.8% |
| Jungjun_AI_Agent | 16 | 50.0% | 75.0% | 100.0% | 71.9% | 100.0% | 71.9% |
| KI_En_ChatGPT_Praktische_Gids | 16 | 56.2% | 62.5% | 93.8% | 78.1% | 100.0% | 59.4% |
| Kimothi_RAG_Guide | 16 | 62.5% | 68.8% | 93.8% | 78.1% | 90.6% | 71.9% |
| Nagasubramanian_Agentic_AI | 16 | 50.0% | 50.0% | 81.2% | 68.8% | 100.0% | 65.6% |
| PCWorld_July_2025 | 16 | 37.5% | 100.0% | 100.0% | 87.5% | 96.9% | 84.4% |
| Python_Cookbook | 16 | 62.5% | 81.2% | 87.5% | 84.4% | 81.2% | 81.2% |
| Python_Distilled | 16 | 93.8% | 100.0% | 100.0% | 93.8% | 93.8% | 87.5% |
| Raieli_AI_Agents | 16 | 50.0% | 75.0% | 93.8% | 84.4% | 100.0% | 68.8% |
| Recent_Trends_in_Transportation | 16 | 56.2% | 93.8% | 100.0% | 81.2% | 71.9% | 75.0% |
| Sekar_MCP_Standard | 16 | 56.2% | 68.8% | 100.0% | 81.2% | 100.0% | 68.8% |

## 4. Weakest 15 (query, top-1) pairs — v2.10.x defect candidates

- **S0203.Q1** total=0/6 (r=0, f=0, faith=0)
  - Query: 'What is the cost associated with the components for generating electricity on a ship?'
  - Gold doc: `Earthship_Vol1` (chunk `aa11d5ea2275_020_text_d5102ec0`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=20 score=0.52918
  - Judge rationale: The chunk is not relevant to the cost of components for generating electricity on a ship, has severe format issues, and does not provide a correct or self-contained answer.
- **S0018.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How can bridging internal silos benefit an organization?'
  - Gold doc: `Devlin_LLM_Agents` (chunk `5b915c809145_185_text_f8cf71aa`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=185 score=0.390652
  - Judge rationale: The chunk does not address the benefits of bridging internal silos and has repeated words, making it neither relevant nor self-contained for the user's query.
- **S0081.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: "How is the specific color of a code lock's backlight indicated in its designation?"
  - Gold doc: `Integra_manual` (chunk `9f3ade9d82e0_046_text_d41ede54`)
  - Top-1: `integra_u_en.pdf` p=47 score=0.621279
  - Judge rationale: The chunk discusses the function of LED colors on a code lock but does not answer how the specific color of a backlight is indicated in its designation, and it has minor formatting issues with odd commas and spaces.
- **S0098.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models are compatible with the oliefilter having art_nr_merk 61551?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_008_text_6eb2d7fc`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=6 score=0.685782
  - Judge rationale: The retrieved chunk does not answer the query about car models compatible with art_nr_merk 61551 and contains minor formatting issues.
- **S0120.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What are the main challenges for commercializing HFCVs?'
  - Gold doc: `Hybrid_electric_vehicles` (chunk `2baf312fdd78_009_text_d7e30e7a`)
  - Top-1: `A_comprehensive_review_on_hybrid_electri.pdf` p=19 score=0.451316
  - Judge rationale: The retrieved chunk discusses challenges related to HEVs and BEVs, not HFCVs, and is slightly truncated at the end.
- **S0151.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the price of the oliefilter for BMW 316 i 318 i (E30)?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_009_text_4518215c`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=10 score=0.69524
  - Judge rationale: The chunk does not provide the price for the oliefilter of BMW 316 i 318 i (E30), and while the format is readable, it's a list with minor issues and would mislead the user on the specific query asked.
- **S0194.Q2** total=1/6 (r=1, f=0, faith=0)
  - Query: "How does the step response differ between 7-triangular input MF's and 7-singleton output MF's in a fuzzy-PD controller?"
  - Gold doc: `Greenhouse_Design` (chunk `8b79e9dca3ae_186_text_64a0ee65`)
  - Top-1: `Greenhouse Design and Control by Pedro Ponce.pdf` p=15 score=0.694928
  - Judge rationale: The chunk mentions the step responses but is severely truncated and lacks coherent structure, making it hard to understand and not self-contained for a correct answer.
- **S0203.Q2** total=1/6 (r=1, f=0, faith=0)
  - Query: 'How does the emission of sunlight contribute to reducing lighting costs?'
  - Gold doc: `Earthship_Vol1` (chunk `aa11d5ea2275_020_text_d5102ec0`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=20 score=0.505442
  - Judge rationale: The chunk is on the same topic but does not directly answer the query, and it has severe formatting issues making it unreadable and misleading.
- **S0238.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What are the page numbers for challenges related to storage and retrieval mechanisms?'
  - Gold doc: `Cronin_GenAI_Models` (chunk `0054f66093d6_634_text_79d7449b_s1_o2`)
  - Top-1: `A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf` p=157 score=0.490946
  - Judge rationale: The retrieved chunk discusses strategies for retrieval and their challenges but does not provide the specific page numbers for challenges related to storage and retrieval mechanisms as requested; the format is a partially readable table with some truncation.
- **S0239.Q1** total=1/6 (r=1, f=0, faith=0)
  - Query: 'What type of shades are recommended for windows in very cold climates?'
  - Gold doc: `Earthship_Vol1` (chunk `aa11d5ea2275_195_text_93149928`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=195 score=0.537501
  - Judge rationale: The retrieved chunk is on the same topic but does not clearly answer the query, and it is severely truncated and garbled, making it unreadable and misleading.
- **S0247.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What does a biomedical agent use in this context?'
  - Gold doc: `Devlin_LLM_Agents` (chunk `5b915c809145_261_text_5cc7d204`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=261 score=0.544402
  - Judge rationale: The retrieved chunk does not provide any information about what a biomedical agent uses and contains repeated words, making it hard to read and understand.
- **S0023.Q2** total=2/6 (r=0, f=2, faith=0)
  - Query: 'How does the server reveal its user ID and why is this a security risk?'
  - Gold doc: `Sekar_MCP_Standard` (chunk `47bcf7e2f91b_202_text_662416ae`)
  - Top-1: `Sekar S. The MCP Standard. A Developer's Guide..Building Universal AI Tools 2026.pdf` p=116 score=0.541686
  - Judge rationale: The retrieved chunk discusses secure user input handling and does not address how a server reveals its user ID or the associated security risk.
- **S0028.Q2** total=2/6 (r=0, f=2, faith=0)
  - Query: 'What information is needed for the payment date and amount?'
  - Gold doc: `Form_betwistingsformulier` (chunk `c33dc178a685_001_text_9add1d0e`)
  - Top-1: `0013_140302111325_001.pdf` p=1 score=0.471701
  - Judge rationale: The retrieved chunk does not provide any information related to the payment date and amount, it only describes the general structure of an invoice document.
- **S0038.Q1** total=2/6 (r=0, f=2, faith=0)
  - Query: 'How does the Power Split Device in LEXUS HEVs function?'
  - Gold doc: `Recent_Trends_in_Transportation` (chunk `fb04e9808444_003_text_c4244741`)
  - Top-1: `Recent_Trends_in_Transportation_Technolo.pdf` p=3 score=0.684642
  - Judge rationale: The retrieved chunk discusses the battery specifications of a Toyota Prius, which is not relevant to the function of the Power Split Device in LEXUS HEVs.
- **S0038.Q2** total=2/6 (r=0, f=2, faith=0)
  - Query: 'What is the voltage of the battery pack after Booster changes it?'
  - Gold doc: `Recent_Trends_in_Transportation` (chunk `fb04e9808444_003_text_c4244741`)
  - Top-1: `Recent_Trends_in_Transportation_Technolo.pdf` p=3 score=0.429857
  - Judge rationale: The retrieved chunk discusses the voltage of the Toyota Prius battery pack, not the voltage after Booster changes it, making it irrelevant and potentially misleading to the user query.

## 5. Methodology

- Sampled 259 text chunks (≥ 150 chars, ≤ 40% code-like lines, no advertisement keywords). Stratified across the 34-doc canonical corpus.
- Each chunk → 2 queries generated by `qwen-max` (temperature 0.3).
- Each query → top-5 retrieved from `mmrag_v2_8__qwen3_dashscope` via `dashscope` provider, model `text-embedding-v4`.
- Each top-1 chunk → graded by `qwen-max` (temperature 0.0) on relevance / format / faithfulness, each 0-2.
- Gold passage is shown to the judge for context; the judge is instructed NOT to penalize a different-chunk same-document retrieval.

## 6. Revision log

| Date | Change |
|---|---|
| 2026-05-21 | Initial v2.10.0-rc1 soak snapshot. |