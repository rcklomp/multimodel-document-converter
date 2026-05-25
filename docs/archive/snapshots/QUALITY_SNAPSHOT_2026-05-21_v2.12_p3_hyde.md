# Quality Snapshot 2026-05-21 — SOAK (mmrag_v2_8__qwen3_dashscope)

> **Status:** synthetic-soak report.
> Source: `output/soak/v2.12_p3_hyde/work.jsonl`.
> Judge: Dashscope `qwen-max`. Generator: `qwen-max`. Embedder: `text-embedding-v4` (provider=dashscope). Collection: `mmrag_v2_8__qwen3_dashscope`. Reranker: `omlx`.
> No QA threshold; this snapshot is informational.

## 1. Corpus summary

- Sampled chunks: **259** across 33 docs.
- Queries generated: **518**.
- Queries judged: **518/518** (100.0%).

## 2. Headline metrics

| Metric | Value |
|---|---:|
| Recall@1 (gold chunk_id is top-1) | 68.3% (354/518) |
| Recall@5 (gold chunk_id in top-5) | 90.2% (467/518) |
| Recall@5 (gold doc_id in top-5)   | 98.5% (510/518) |
| Relevance score                   | 82.0% (850/1036) |
| Format score                      | 87.7% (909/1036) |
| Faithfulness score                | 73.5% (761/1036) |

## 3. Per-document metrics

| Doc | Queries | R@1 | R@5 (chunk) | R@5 (doc) | Relevance | Format | Faith |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIOS_LLM_Agent_Operating_System | 16 | 81.2% | 81.2% | 87.5% | 84.4% | 96.9% | 75.0% |
| ATZ_Elektronik_German | 16 | 75.0% | 87.5% | 93.8% | 71.9% | 84.4% | 56.2% |
| A_comprehensive_review_on_hybrid_electri | 16 | 75.0% | 100.0% | 100.0% | 71.9% | 81.2% | 68.8% |
| Adedeji_GenAI_Google_Cloud | 16 | 87.5% | 100.0% | 100.0% | 87.5% | 90.6% | 81.2% |
| ArcGIS_Python_Cookbook | 16 | 50.0% | 81.2% | 93.8% | 84.4% | 93.8% | 78.1% |
| Ayeva_Python_Patterns | 16 | 87.5% | 93.8% | 100.0% | 90.6% | 90.6% | 78.1% |
| Bourne_RAG_2024 | 16 | 62.5% | 81.2% | 100.0% | 81.2% | 90.6% | 62.5% |
| CarOK_voorraadtelling | 16 | 31.2% | 68.8% | 100.0% | 46.9% | 53.1% | 25.0% |
| ChatGPT_Praktijk_handboek | 16 | 56.2% | 87.5% | 100.0% | 68.8% | 100.0% | 65.6% |
| Chaubal_PyTorch_Projects | 16 | 75.0% | 93.8% | 100.0% | 87.5% | 100.0% | 78.1% |
| Combat_Aircraft_August_2025 | 16 | 87.5% | 100.0% | 100.0% | 90.6% | 71.9% | 87.5% |
| Cronin_GenAI_Models | 16 | 56.2% | 81.2% | 93.8% | 75.0% | 96.9% | 71.9% |
| Devlin_LLM_Agents | 16 | 87.5% | 100.0% | 100.0% | 71.9% | 71.9% | 65.6% |
| Earthship_Vol1 | 16 | 87.5% | 100.0% | 100.0% | 71.9% | 59.4% | 65.6% |
| Firearms | 16 | 56.2% | 87.5% | 100.0% | 84.4% | 71.9% | 75.0% |
| Fluent_Python | 16 | 62.5% | 100.0% | 100.0% | 78.1% | 100.0% | 68.8% |
| Form_betwistingsformulier | 6 | 50.0% | 66.7% | 83.3% | 58.3% | 91.7% | 50.0% |
| Greenhouse_Design | 16 | 62.5% | 93.8% | 100.0% | 90.6% | 78.1% | 78.1% |
| Hao_ML_Platform | 16 | 50.0% | 75.0% | 93.8% | 81.2% | 96.9% | 71.9% |
| HarryPotter_and_the_Sorcerers_Stone | 16 | 56.2% | 100.0% | 100.0% | 90.6% | 90.6% | 81.2% |
| Hybrid_electric_vehicles | 16 | 93.8% | 93.8% | 100.0% | 84.4% | 93.8% | 78.1% |
| IRJET_Modeling_of_Solar_PV | 16 | 81.2% | 81.2% | 100.0% | 87.5% | 84.4% | 81.2% |
| Integra_manual | 16 | 62.5% | 87.5% | 100.0% | 75.0% | 84.4% | 75.0% |
| Jungjun_AI_Agent | 16 | 50.0% | 93.8% | 100.0% | 87.5% | 100.0% | 84.4% |
| KI_En_ChatGPT_Praktische_Gids | 16 | 75.0% | 87.5% | 100.0% | 75.0% | 100.0% | 62.5% |
| Kimothi_RAG_Guide | 16 | 81.2% | 87.5% | 100.0% | 90.6% | 96.9% | 81.2% |
| Nagasubramanian_Agentic_AI | 16 | 81.2% | 100.0% | 100.0% | 96.9% | 100.0% | 90.6% |
| PCWorld_July_2025 | 16 | 43.8% | 100.0% | 100.0% | 87.5% | 96.9% | 84.4% |
| Python_Cookbook | 16 | 75.0% | 93.8% | 93.8% | 90.6% | 78.1% | 81.2% |
| Python_Distilled | 16 | 87.5% | 87.5% | 100.0% | 87.5% | 96.9% | 78.1% |
| Raieli_AI_Agents | 16 | 43.8% | 87.5% | 100.0% | 81.2% | 96.9% | 75.0% |
| Recent_Trends_in_Transportation | 16 | 68.8% | 100.0% | 100.0% | 93.8% | 65.6% | 78.1% |
| Sekar_MCP_Standard | 16 | 62.5% | 81.2% | 100.0% | 87.5% | 93.8% | 75.0% |

## 4. Weakest 15 (query, top-1) pairs — v2.10.x defect candidates

- **S0203.Q1** total=0/6 (r=0, f=0, faith=0)
  - Query: 'What is the cost associated with the components for generating electricity on a ship?'
  - Gold doc: `Earthship_Vol1` (chunk `aa11d5ea2275_020_text_d5102ec0`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=20 score=0.462516
  - Judge rationale: The retrieved chunk is not relevant to the user query, has severe format issues, and does not provide a correct or self-contained answer.
- **S0018.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How can bridging internal silos benefit an organization?'
  - Gold doc: `Devlin_LLM_Agents` (chunk `5b915c809145_185_text_f8cf71aa`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=185 score=0.0
  - Judge rationale: The chunk does not answer the query about benefits of bridging internal silos and has repeated words, making it confusing and uninformative.
- **S0080.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How does the script concatenate strings to form the countdown?'
  - Gold doc: `Python_Cookbook` (chunk `0326dff0bbb4_416_text_82f981d0`)
  - Top-1: `Python Distilled David M. Beazley 2022.pdf` p=867 score=0.645867
  - Judge rationale: The retrieved chunk does not provide any information on string concatenation for a countdown and is only described as a code snippet without showing the actual code or explanation, making it irrelevant and not self-contained for the user's query.
- **S0081.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: "What determines the color version of the code lock's backlight?"
  - Gold doc: `Integra_manual` (chunk `9f3ade9d82e0_046_text_d41ede54`)
  - Top-1: `integra_u_en.pdf` p=47 score=0.637272
  - Judge rationale: The retrieved chunk discusses the function of different LED colors on the code lock but does not address what determines the color version of the backlight, and it is slightly messy in format.
- **S0081.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: "How is the specific color of a code lock's backlight indicated in its designation?"
  - Gold doc: `Integra_manual` (chunk `9f3ade9d82e0_046_text_d41ede54`)
  - Top-1: `integra_u_en.pdf` p=47 score=0.572043
  - Judge rationale: The chunk does not answer the query about how a code lock's backlight color is indicated in its designation and instead provides information on LED functions, with minor formatting issues.
- **S0095.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What components can be removed after ejecting downward?'
  - Gold doc: `Firearms` (chunk `29f7c8bb7680_156_text_b728ae31`)
  - Top-1: `Firearms.pdf` p=200 score=0.0
  - Judge rationale: The chunk does not answer the query about components to be removed after ejecting downward and has minor formatting issues with truncation and odd spacing.
- **S0098.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models are compatible with the oliefilter having art_nr_merk 61551?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_008_text_6eb2d7fc`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=7 score=0.697529
  - Judge rationale: The chunk does not contain information about the car models compatible with art_nr_merk 61551, and while the format is readable, it's not relevant to the query, making it misleading for the user.
- **S0115.Q1** total=1/6 (r=1, f=0, faith=0)
  - Query: 'What are the steps involved in building an MCP client?'
  - Gold doc: `Sekar_MCP_Standard` (chunk `47bcf7e2f91b_008_text_8317f7f1`)
  - Top-1: `Sekar S. The MCP Standard. A Developer's Guide..Building Universal AI Tools 2026.pdf` p=135 score=0.0
  - Judge rationale: The chunk is about building an MCP client but only shows a part of the code without context, and it's poorly formatted with severe truncation, making it hard to understand and not self-contained.
- **S0125.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models use the oliefilter with Mapco, ink.ex.BTW Titel 2,30?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_007_text_7b2e5fc3`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=10 score=0.756917
  - Judge rationale: The retrieved chunk does not mention the specific oliefilter with Mapco, ink.ex.BTW Titel 2,30 or the car models that use it, and there are minor formatting issues such as truncation at the end.
- **S0151.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the price of the oliefilter for BMW 316 i 318 i (E30)?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_009_text_4518215c`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=6 score=0.570993
  - Judge rationale: The retrieved chunk does not provide the price for the oil filter of BMW 316 i 318 i (E30), and the format has minor issues with truncation and odd whitespace, making it misleading or wrong for the user's query.
- **S0180.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the price of the Luchtfilter Opel 4416403 excluding BTW?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_004_text_9bdca6fe_o2`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=5 score=0.639381
  - Judge rationale: The retrieved chunk does not contain the price for Luchtfilter Opel 4416403, and the format is messy with truncated sentences and odd spacing.
- **S0194.Q2** total=1/6 (r=1, f=0, faith=0)
  - Query: "How does the step response differ between 7-triangular input MF's and 7-singleton output MF's in a fuzzy-PD controller?"
  - Gold doc: `Greenhouse_Design` (chunk `8b79e9dca3ae_186_text_64a0ee65`)
  - Top-1: `Greenhouse Design and Control by Pedro Ponce.pdf` p=15 score=0.688104
  - Judge rationale: The chunk mentions the step responses but is severely truncated and mixed with other unrelated information, making it confusing and not directly answering the specific difference asked for.
- **S0198.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which vehicles are compatible with the Mapco oil filter 61098?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_007_text_672f1c0b`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=8 score=0.72505
  - Judge rationale: The retrieved chunk does not mention the Mapco oil filter 61098 or any compatible vehicles, and it has minor formatting issues with odd whitespace and structure.
- **S0238.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What are the page numbers for challenges related to storage and retrieval mechanisms?'
  - Gold doc: `Cronin_GenAI_Models` (chunk `0054f66093d6_634_text_79d7449b_s1_o2`)
  - Top-1: `A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf` p=157 score=0.447723
  - Judge rationale: The retrieved chunk does not address the page numbers for challenges related to storage and retrieval mechanisms, and it is a table with minor truncation issues, making it irrelevant and misleading for the user's query.
- **S0239.Q1** total=1/6 (r=1, f=0, faith=0)
  - Query: 'What type of shades are recommended for windows in very cold climates?'
  - Gold doc: `Earthship_Vol1` (chunk `aa11d5ea2275_195_text_93149928`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=195 score=0.520324
  - Judge rationale: The chunk is on the same topic but does not clearly answer the query due to severe truncation and garbled text, making it unreadable and unhelpful.

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