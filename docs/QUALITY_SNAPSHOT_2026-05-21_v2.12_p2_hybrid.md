# Quality Snapshot 2026-05-21 — SOAK (mmrag_v2_8__qwen3_dashscope)

> **Status:** synthetic-soak report.
> Source: `output/soak/v2.12_p2_hybrid/work.jsonl`.
> Judge: Dashscope `qwen-max`. Generator: `qwen-max`. Embedder: `text-embedding-v4` (provider=dashscope). Collection: `mmrag_v2_8__qwen3_dashscope`. Reranker: `omlx`.
> No QA threshold; this snapshot is informational.

## 1. Corpus summary

- Sampled chunks: **259** across 33 docs.
- Queries generated: **518**.
- Queries judged: **518/518** (100.0%).

## 2. Headline metrics

| Metric | Value |
|---|---:|
| Recall@1 (gold chunk_id is top-1) | 67.8% (351/518) |
| Recall@5 (gold chunk_id in top-5) | 90.2% (467/518) |
| Recall@5 (gold doc_id in top-5)   | 98.6% (511/518) |
| Relevance score                   | 82.1% (851/1036) |
| Format score                      | 88.4% (916/1036) |
| Faithfulness score                | 72.6% (752/1036) |

## 3. Per-document metrics

| Doc | Queries | R@1 | R@5 (chunk) | R@5 (doc) | Relevance | Format | Faith |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIOS_LLM_Agent_Operating_System | 16 | 87.5% | 87.5% | 93.8% | 87.5% | 93.8% | 68.8% |
| ATZ_Elektronik_German | 16 | 75.0% | 87.5% | 93.8% | 71.9% | 87.5% | 59.4% |
| A_comprehensive_review_on_hybrid_electri | 16 | 75.0% | 100.0% | 100.0% | 75.0% | 81.2% | 71.9% |
| Adedeji_GenAI_Google_Cloud | 16 | 75.0% | 100.0% | 100.0% | 90.6% | 90.6% | 78.1% |
| ArcGIS_Python_Cookbook | 16 | 50.0% | 81.2% | 93.8% | 84.4% | 93.8% | 78.1% |
| Ayeva_Python_Patterns | 16 | 87.5% | 93.8% | 100.0% | 90.6% | 90.6% | 78.1% |
| Bourne_RAG_2024 | 16 | 50.0% | 75.0% | 100.0% | 78.1% | 93.8% | 59.4% |
| CarOK_voorraadtelling | 16 | 31.2% | 68.8% | 100.0% | 53.1% | 62.5% | 37.5% |
| ChatGPT_Praktijk_handboek | 16 | 56.2% | 81.2% | 100.0% | 71.9% | 100.0% | 68.8% |
| Chaubal_PyTorch_Projects | 16 | 75.0% | 93.8% | 100.0% | 87.5% | 100.0% | 78.1% |
| Combat_Aircraft_August_2025 | 16 | 87.5% | 100.0% | 100.0% | 90.6% | 75.0% | 87.5% |
| Cronin_GenAI_Models | 16 | 62.5% | 81.2% | 93.8% | 78.1% | 96.9% | 75.0% |
| Devlin_LLM_Agents | 16 | 87.5% | 100.0% | 100.0% | 71.9% | 71.9% | 68.8% |
| Earthship_Vol1 | 16 | 87.5% | 93.8% | 100.0% | 71.9% | 62.5% | 65.6% |
| Firearms | 16 | 56.2% | 87.5% | 100.0% | 81.2% | 68.8% | 68.8% |
| Fluent_Python | 16 | 50.0% | 100.0% | 100.0% | 78.1% | 100.0% | 75.0% |
| Form_betwistingsformulier | 6 | 33.3% | 66.7% | 83.3% | 41.7% | 100.0% | 33.3% |
| Greenhouse_Design | 16 | 56.2% | 100.0% | 100.0% | 90.6% | 81.2% | 75.0% |
| Hao_ML_Platform | 16 | 50.0% | 75.0% | 100.0% | 78.1% | 93.8% | 62.5% |
| HarryPotter_and_the_Sorcerers_Stone | 16 | 56.2% | 100.0% | 100.0% | 93.8% | 87.5% | 81.2% |
| Hybrid_electric_vehicles | 16 | 93.8% | 93.8% | 100.0% | 84.4% | 93.8% | 75.0% |
| IRJET_Modeling_of_Solar_PV | 16 | 75.0% | 81.2% | 93.8% | 81.2% | 84.4% | 65.6% |
| Integra_manual | 16 | 62.5% | 87.5% | 100.0% | 75.0% | 87.5% | 75.0% |
| Jungjun_AI_Agent | 16 | 62.5% | 93.8% | 100.0% | 84.4% | 100.0% | 84.4% |
| KI_En_ChatGPT_Praktische_Gids | 16 | 75.0% | 87.5% | 100.0% | 78.1% | 100.0% | 62.5% |
| Kimothi_RAG_Guide | 16 | 75.0% | 81.2% | 100.0% | 87.5% | 93.8% | 75.0% |
| Nagasubramanian_Agentic_AI | 16 | 87.5% | 100.0% | 100.0% | 96.9% | 100.0% | 87.5% |
| PCWorld_July_2025 | 16 | 37.5% | 100.0% | 100.0% | 84.4% | 96.9% | 78.1% |
| Python_Cookbook | 16 | 75.0% | 93.8% | 93.8% | 90.6% | 81.2% | 84.4% |
| Python_Distilled | 16 | 93.8% | 93.8% | 100.0% | 90.6% | 93.8% | 81.2% |
| Raieli_AI_Agents | 16 | 50.0% | 87.5% | 100.0% | 84.4% | 96.9% | 75.0% |
| Recent_Trends_in_Transportation | 16 | 68.8% | 100.0% | 100.0% | 93.8% | 68.8% | 84.4% |
| Sekar_MCP_Standard | 16 | 68.8% | 87.5% | 100.0% | 87.5% | 96.9% | 71.9% |

## 4. Weakest 15 (query, top-1) pairs — v2.10.x defect candidates

- **S0203.Q1** total=0/6 (r=0, f=0, faith=0)
  - Query: 'What is the cost associated with the components for generating electricity on a ship?'
  - Gold doc: `Earthship_Vol1` (chunk `aa11d5ea2275_020_text_d5102ec0`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=20 score=0.529192
  - Judge rationale: The retrieved chunk is not relevant to the user query, has severe format issues, and does not provide a correct or self-contained answer.
- **S0018.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How can bridging internal silos benefit an organization?'
  - Gold doc: `Devlin_LLM_Agents` (chunk `5b915c809145_185_text_f8cf71aa`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=185 score=0.390652
  - Judge rationale: The chunk does not answer the query about benefits of bridging internal silos and has repeated words making it hard to read and uninformative.
- **S0081.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: "What determines the color version of the code lock's backlight?"
  - Gold doc: `Integra_manual` (chunk `9f3ade9d82e0_046_text_d41ede54`)
  - Top-1: `integra_u_en.pdf` p=47 score=0.636508
  - Judge rationale: The chunk discusses the function of different LED colors on a code lock but does not address what determines the color version of the backlight, and it has minor format issues with odd punctuation and structure.
- **S0081.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: "How is the specific color of a code lock's backlight indicated in its designation?"
  - Gold doc: `Integra_manual` (chunk `9f3ade9d82e0_046_text_d41ede54`)
  - Top-1: `integra_u_en.pdf` p=47 score=0.621288
  - Judge rationale: The chunk does not address the query about color designation in code lock's backlight and is a list with minor formatting issues, making it irrelevant and unfaithful.
- **S0098.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models are compatible with the oliefilter having art_nr_merk 61551?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_008_text_6eb2d7fc`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=7 score=0.662129
  - Judge rationale: The retrieved chunk does not contain information about the car models compatible with the oliefilter having art_nr_merk 61551, and while the format is readable, it is not relevant to the user's query, leading to a misleading answer.
- **S0125.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models use the oliefilter with Mapco, ink.ex.BTW Titel 2,30?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_007_text_7b2e5fc3`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=8 score=0.734098
  - Judge rationale: The retrieved chunk does not answer the user query about car models using a specific oliefilter and contains minor formatting issues, making it misleading for the user's question.
- **S0180.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the price of the Luchtfilter Opel 4416403 excluding BTW?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_004_text_9bdca6fe_o2`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=5 score=0.649907
  - Judge rationale: The retrieved chunk does not contain the price for the Luchtfilter Opel 4416403 and is poorly formatted with some truncation and odd whitespace, making it neither relevant nor faithful to the user's query.
- **S0181.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What software was used for modeling and simulating the system?'
  - Gold doc: `IRJET_Modeling_of_Solar_PV` (chunk `b670f5359e9c_006_text_7d9ffb90`)
  - Top-1: `Hybrid_electric_vehicles_and_their_challenges.pdf` p=8 score=0.0
  - Judge rationale: The retrieved chunk discusses a different system and does not mention the software used for modeling and simulating, it has minor formatting issues with truncation, and provides no correct answer to the user's query.
- **S0194.Q2** total=1/6 (r=1, f=0, faith=0)
  - Query: "How does the step response differ between 7-triangular input MF's and 7-singleton output MF's in a fuzzy-PD controller?"
  - Gold doc: `Greenhouse_Design` (chunk `8b79e9dca3ae_186_text_64a0ee65`)
  - Top-1: `Greenhouse Design and Control by Pedro Ponce.pdf` p=15 score=0.69494
  - Judge rationale: The chunk mentions the step response and membership functions but is poorly formatted and does not provide a clear, self-contained answer to the query.
- **S0198.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which vehicles are compatible with the Mapco oil filter 61098?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_007_text_672f1c0b`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=8 score=0.67395
  - Judge rationale: The retrieved chunk does not provide information about the compatibility of Mapco oil filter 61098 with any vehicles and contains minor formatting issues.
- **S0238.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What are the page numbers for challenges related to storage and retrieval mechanisms?'
  - Gold doc: `Cronin_GenAI_Models` (chunk `0054f66093d6_634_text_79d7449b_s1_o2`)
  - Top-1: `A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf` p=157 score=0.491065
  - Judge rationale: The retrieved chunk does not answer the user query about page numbers and instead provides a table on retrieval strategies, with minor formatting issues and is misleading as it does not contain the requested information.
- **S0239.Q1** total=1/6 (r=1, f=0, faith=0)
  - Query: 'What type of shades are recommended for windows in very cold climates?'
  - Gold doc: `Earthship_Vol1` (chunk `aa11d5ea2275_195_text_93149928`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=195 score=0.537433
  - Judge rationale: The retrieved chunk is on the same topic but does not clearly answer the query due to severe truncation and formatting issues, making it unreadable and unhelpful.
- **S0247.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What does a biomedical agent use in this context?'
  - Gold doc: `Devlin_LLM_Agents` (chunk `5b915c809145_261_text_5cc7d204`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=261 score=0.544459
  - Judge rationale: The chunk does not provide any information about what a biomedical agent uses, and it contains repetitive words and phrases making it hard to read, although no severe formatting issues are present.
- **S0007.Q1** total=2/6 (r=0, f=2, faith=0)
  - Query: 'What is the requirement for submitting manuscripts to the editorial office?'
  - Gold doc: `ATZ_Elektronik_German` (chunk `6fccda8bd625_006_text_762320b5`)
  - Top-1: `PCWorld_July_2025_USA.pdf` p=5 score=0.0
  - Judge rationale: The retrieved chunk provides contact information for an editorial office but does not address the requirements for submitting manuscripts as requested in the user query.
- **S0028.Q2** total=2/6 (r=0, f=2, faith=0)
  - Query: 'What information is needed for the payment date and amount?'
  - Gold doc: `Form_betwistingsformulier` (chunk `c33dc178a685_001_text_9add1d0e`)
  - Top-1: `0013_140302111325_001.pdf` p=1 score=0.471964
  - Judge rationale: The retrieved chunk does not provide any information about the payment date and amount, and instead describes an invoice document's structure, which is not relevant to the user's query.

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