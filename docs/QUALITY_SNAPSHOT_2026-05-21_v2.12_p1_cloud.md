# Quality Snapshot 2026-05-21 — SOAK (mmrag_v2_8__qwen3_dashscope)

> **Status:** synthetic-soak report.
> Source: `output/soak/v2.12_p1_cloud/work.jsonl`.
> Judge: Dashscope `qwen-max`. Generator: `qwen-max`. Embedder: `text-embedding-v4` (provider=dashscope). Collection: `mmrag_v2_8__qwen3_dashscope`. Reranker: `dashscope`.
> No QA threshold; this snapshot is informational.

## 1. Corpus summary

- Sampled chunks: **259** across 33 docs.
- Queries generated: **518**.
- Queries judged: **518/518** (100.0%).

## 2. Headline metrics

| Metric | Value |
|---|---:|
| Recall@1 (gold chunk_id is top-1) | 53.9% (279/518) |
| Recall@5 (gold chunk_id in top-5) | 66.8% (346/518) |
| Recall@5 (gold doc_id in top-5)   | 91.7% (475/518) |
| Relevance score                   | 74.5% (772/1036) |
| Format score                      | 89.5% (927/1036) |
| Faithfulness score                | 64.2% (665/1036) |

## 3. Per-document metrics

| Doc | Queries | R@1 | R@5 (chunk) | R@5 (doc) | Relevance | Format | Faith |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIOS_LLM_Agent_Operating_System | 16 | 75.0% | 75.0% | 87.5% | 78.1% | 93.8% | 68.8% |
| ATZ_Elektronik_German | 16 | 81.2% | 81.2% | 93.8% | 81.2% | 87.5% | 65.6% |
| A_comprehensive_review_on_hybrid_electri | 16 | 62.5% | 68.8% | 100.0% | 75.0% | 81.2% | 62.5% |
| Adedeji_GenAI_Google_Cloud | 16 | 62.5% | 75.0% | 100.0% | 78.1% | 93.8% | 71.9% |
| ArcGIS_Python_Cookbook | 16 | 31.2% | 50.0% | 81.2% | 68.8% | 90.6% | 62.5% |
| Ayeva_Python_Patterns | 16 | 68.8% | 81.2% | 93.8% | 75.0% | 90.6% | 62.5% |
| Bourne_RAG_2024 | 16 | 50.0% | 50.0% | 81.2% | 68.8% | 96.9% | 56.2% |
| CarOK_voorraadtelling | 16 | 31.2% | 43.8% | 100.0% | 56.2% | 53.1% | 40.6% |
| ChatGPT_Praktijk_handboek | 16 | 56.2% | 56.2% | 81.2% | 81.2% | 100.0% | 68.8% |
| Chaubal_PyTorch_Projects | 16 | 50.0% | 56.2% | 87.5% | 75.0% | 100.0% | 65.6% |
| Combat_Aircraft_August_2025 | 16 | 81.2% | 93.8% | 100.0% | 84.4% | 71.9% | 78.1% |
| Cronin_GenAI_Models | 16 | 43.8% | 56.2% | 68.8% | 68.8% | 100.0% | 68.8% |
| Devlin_LLM_Agents | 16 | 56.2% | 62.5% | 75.0% | 65.6% | 78.1% | 56.2% |
| Earthship_Vol1 | 16 | 75.0% | 75.0% | 100.0% | 65.6% | 65.6% | 59.4% |
| Firearms | 16 | 56.2% | 75.0% | 100.0% | 78.1% | 75.0% | 59.4% |
| Fluent_Python | 16 | 56.2% | 81.2% | 93.8% | 75.0% | 100.0% | 71.9% |
| Form_betwistingsformulier | 6 | 66.7% | 66.7% | 83.3% | 66.7% | 100.0% | 66.7% |
| Greenhouse_Design | 16 | 37.5% | 87.5% | 100.0% | 84.4% | 81.2% | 75.0% |
| Hao_ML_Platform | 16 | 37.5% | 62.5% | 100.0% | 75.0% | 100.0% | 62.5% |
| HarryPotter_and_the_Sorcerers_Stone | 16 | 56.2% | 81.2% | 100.0% | 75.0% | 84.4% | 59.4% |
| Hybrid_electric_vehicles | 16 | 75.0% | 81.2% | 100.0% | 71.9% | 90.6% | 65.6% |
| IRJET_Modeling_of_Solar_PV | 16 | 68.8% | 75.0% | 93.8% | 81.2% | 93.8% | 75.0% |
| Integra_manual | 16 | 37.5% | 62.5% | 100.0% | 68.8% | 87.5% | 56.2% |
| Jungjun_AI_Agent | 16 | 25.0% | 43.8% | 81.2% | 59.4% | 100.0% | 53.1% |
| KI_En_ChatGPT_Praktische_Gids | 16 | 43.8% | 50.0% | 100.0% | 68.8% | 100.0% | 53.1% |
| Kimothi_RAG_Guide | 16 | 31.2% | 43.8% | 68.8% | 75.0% | 100.0% | 56.2% |
| Nagasubramanian_Agentic_AI | 16 | 37.5% | 37.5% | 87.5% | 65.6% | 100.0% | 59.4% |
| PCWorld_July_2025 | 16 | 37.5% | 81.2% | 100.0% | 90.6% | 93.8% | 81.2% |
| Python_Cookbook | 16 | 62.5% | 62.5% | 75.0% | 71.9% | 84.4% | 71.9% |
| Python_Distilled | 16 | 81.2% | 81.2% | 100.0% | 90.6% | 93.8% | 75.0% |
| Raieli_AI_Agents | 16 | 56.2% | 75.0% | 93.8% | 81.2% | 96.9% | 71.9% |
| Recent_Trends_in_Transportation | 16 | 56.2% | 81.2% | 93.8% | 78.1% | 75.0% | 65.6% |
| Sekar_MCP_Standard | 16 | 37.5% | 50.0% | 100.0% | 75.0% | 100.0% | 53.1% |

## 4. Weakest 15 (query, top-1) pairs — v2.10.x defect candidates

- **S0203.Q1** total=0/6 (r=0, f=0, faith=0)
  - Query: 'What is the cost associated with the components for generating electricity on a ship?'
  - Gold doc: `Earthship_Vol1` (chunk `aa11d5ea2275_020_text_d5102ec0`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=20 score=0.52914
  - Judge rationale: The retrieved chunk is not relevant to the cost of components for generating electricity on a ship, has severe formatting issues, and does not provide a correct or self-contained answer.
- **S0018.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How can bridging internal silos benefit an organization?'
  - Gold doc: `Devlin_LLM_Agents` (chunk `5b915c809145_185_text_f8cf71aa`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=185 score=0.390605
  - Judge rationale: The chunk does not answer the query about the benefits of bridging internal silos and has repeated words, making it confusing and misleading.
- **S0055.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How does the electric motor in hybrid cars contribute to better fuel economy?'
  - Gold doc: `Recent_Trends_in_Transportation` (chunk `fb04e9808444_004_text_f1744558`)
  - Top-1: `Recent_Trends_in_Transportation_Technolo.pdf` p=5 score=0.544366
  - Judge rationale: The retrieved chunk is a list of references and does not provide any direct information about how electric motors in hybrid cars contribute to better fuel economy, making it irrelevant and not self-contained for the answer.
- **S0080.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How does the script concatenate strings to form the countdown?'
  - Gold doc: `Python_Cookbook` (chunk `0326dff0bbb4_416_text_82f981d0`)
  - Top-1: `Python Distilled David M. Beazley 2022.pdf` p=963 score=0.700745
  - Judge rationale: The retrieved chunk does not provide any information on how strings are concatenated in the script, and it is a vague description rather than a code snippet or clear explanation.
- **S0081.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: "What determines the color version of the code lock's backlight?"
  - Gold doc: `Integra_manual` (chunk `9f3ade9d82e0_046_text_d41ede54`)
  - Top-1: `integra_u_en.pdf` p=47 score=0.63644
  - Judge rationale: The retrieved chunk discusses the functions of different LED colors but does not address how the color version of the code lock's backlight is determined, and its format has minor issues with odd spacing and structure.
- **S0081.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: "How is the specific color of a code lock's backlight indicated in its designation?"
  - Gold doc: `Integra_manual` (chunk `9f3ade9d82e0_046_text_d41ede54`)
  - Top-1: `integra_u_en.pdf` p=47 score=0.621336
  - Judge rationale: The retrieved chunk discusses the function of different LED colors on a code lock but does not answer how the specific color of a backlight is indicated in its designation, and it has minor formatting issues with odd spacing and punctuation.
- **S0098.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models are compatible with the oliefilter having art_nr_merk 61551?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_008_text_6eb2d7fc`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=6 score=0.68579
  - Judge rationale: The retrieved chunk does not answer the query about car models compatible with art_nr_merk 61551, and it has minor formatting issues with odd whitespace and truncation, making it misleading for the user's question.
- **S0144.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Why is it not advisable to expand a single room to make a house?'
  - Gold doc: `Earthship_Vol1` (chunk `aa11d5ea2275_055_text_fc06da69`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=59 score=0.536132
  - Judge rationale: The chunk discusses the placement of rooms for heating purposes and does not address why a single room cannot be expanded to make a house, and it is somewhat truncated and less readable.
- **S0180.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the price of the Luchtfilter Opel 4416403 excluding BTW?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_004_text_9bdca6fe_o2`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=4 score=0.63664
  - Judge rationale: The retrieved chunk does not contain the price for Luchtfilter Opel 4416403 and is part of a table with minor formatting issues, making it neither relevant nor faithful to the query.
- **S0198.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which vehicles are compatible with the Mapco oil filter 61098?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_007_text_672f1c0b`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=8 score=0.67395
  - Judge rationale: The retrieved chunk does not answer the user query about Mapco oil filter 61098 compatibility and instead provides information on different Mapco oil filters; the format is slightly off due to truncation and odd whitespace but is still readable.
- **S0198.Q2** total=1/6 (r=1, f=0, faith=0)
  - Query: 'How much does the oil filter for a Ford Escort cost excluding VAT?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_007_text_672f1c0b`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=8 score=0.561131
  - Judge rationale: The chunk mentions the Ford Escort but does not clearly state the price of the oil filter excluding VAT, and the format is confusing with multiple entries and truncations.
- **S0236.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How do you identify the correct Opel interieurfilter based on the provided information?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_004_text_9bdca6fe`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=5 score=0.651012
  - Judge rationale: The retrieved chunk does not provide information on how to identify the correct Opel interieurfilter and is poorly formatted with incomplete sentences and data, making it misleading or wrong for the user's query.
- **S0239.Q1** total=1/6 (r=1, f=0, faith=0)
  - Query: 'What type of shades are recommended for windows in very cold climates?'
  - Gold doc: `Earthship_Vol1` (chunk `aa11d5ea2275_195_text_93149928`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=198 score=0.613578
  - Judge rationale: The chunk discusses shading in cold climates but is severely truncated and garbled, making it confusing and potentially misleading.
- **S0259.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the difference between instance level and class level lazy initialization?'
  - Gold doc: `Ayeva_Python_Patterns` (chunk `289fd158f828_126_text_813d7b64`)
  - Top-1: `Python Distilled David M. Beazley 2022.pdf` p=332 score=0.578025
  - Judge rationale: The retrieved chunk does not address the difference between instance and class level lazy initialization, and is truncated, making it hard to understand and not self-contained for the question asked.
- **S0020.Q1** total=2/6 (r=0, f=2, faith=0)
  - Query: 'What does the _parse_response method do in the API?'
  - Gold doc: `Jungjun_AI_Agent` (chunk `6afeb55a9449_119_text_14cd977d`)
  - Top-1: `Programming ArcGIS with Python Cookbook.pdf` p=289 score=0.540867
  - Judge rationale: The retrieved chunk discusses making and parsing REST requests in Python, which is unrelated to the _parse_response method of an API as described in the user query.

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