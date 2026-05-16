# Quality Snapshot 2026-05-16 — v2.10.0-rc1 SOAK

> **Status:** synthetic-soak report (Phase 9 candidate).
> Source: `output/soak/v2.10/work.jsonl`.
> Judge: Dashscope `qwen-max`. Generator: `qwen-max`. Embedder: `llava` (Ollama).
> No QA threshold; this snapshot is informational, intended to seed v2.10.x defect candidates.

## 1. Corpus summary

- Sampled chunks: **259** across 33 docs.
- Queries generated: **518**.
- Queries judged: **518/518** (100.0%).

## 2. Headline metrics

| Metric | Value |
|---|---:|
| Recall@1 (gold chunk_id is top-1) | 2.1% (11/518) |
| Recall@5 (gold chunk_id in top-5) | 6.8% (35/518) |
| Recall@5 (gold doc_id in top-5)   | 54.2% (281/518) |
| Relevance score                   | 5.9% (61/1036) |
| Format score                      | 98.3% (1018/1036) |
| Faithfulness score                | 4.7% (49/1036) |

## 3. Per-document metrics

| Doc | Queries | R@1 | R@5 (chunk) | R@5 (doc) | Relevance | Format | Faith |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIOS_LLM_Agent_Operating_System | 16 | 0.0% | 0.0% | 25.0% | 3.1% | 96.9% | 3.1% |
| ATZ_Elektronik_German | 16 | 6.2% | 12.5% | 37.5% | 6.2% | 96.9% | 3.1% |
| A_comprehensive_review_on_hybrid_electri | 16 | 0.0% | 0.0% | 62.5% | 6.2% | 100.0% | 3.1% |
| Adedeji_GenAI_Google_Cloud | 16 | 0.0% | 6.2% | 68.8% | 0.0% | 100.0% | 0.0% |
| ArcGIS_Python_Cookbook | 16 | 6.2% | 12.5% | 75.0% | 25.0% | 100.0% | 21.9% |
| Ayeva_Python_Patterns | 16 | 12.5% | 12.5% | 62.5% | 21.9% | 100.0% | 18.8% |
| Bourne_RAG_2024 | 16 | 0.0% | 0.0% | 0.0% | 12.5% | 100.0% | 9.4% |
| CarOK_voorraadtelling | 16 | 0.0% | 0.0% | 6.2% | 0.0% | 96.9% | 0.0% |
| ChatGPT_Praktijk_handboek | 16 | 0.0% | 0.0% | 12.5% | 0.0% | 100.0% | 0.0% |
| Chaubal_PyTorch_Projects | 16 | 0.0% | 0.0% | 62.5% | 3.1% | 100.0% | 3.1% |
| Combat_Aircraft_August_2025 | 16 | 6.2% | 12.5% | 93.8% | 3.1% | 96.9% | 3.1% |
| Cronin_GenAI_Models | 16 | 0.0% | 6.2% | 37.5% | 6.2% | 100.0% | 6.2% |
| Devlin_LLM_Agents | 16 | 0.0% | 6.2% | 87.5% | 12.5% | 100.0% | 12.5% |
| Earthship_Vol1 | 16 | 6.2% | 12.5% | 75.0% | 12.5% | 100.0% | 6.2% |
| Firearms | 16 | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% |
| Fluent_Python | 16 | 0.0% | 6.2% | 87.5% | 0.0% | 96.9% | 0.0% |
| Form_betwistingsformulier | 6 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| Greenhouse_Design | 16 | 6.2% | 18.8% | 87.5% | 9.4% | 93.8% | 6.2% |
| Hao_ML_Platform | 16 | 0.0% | 0.0% | 43.8% | 0.0% | 100.0% | 0.0% |
| HarryPotter_and_the_Sorcerers_Stone | 16 | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% |
| Hybrid_electric_vehicles | 16 | 12.5% | 25.0% | 68.8% | 9.4% | 96.9% | 9.4% |
| IRJET_Modeling_of_Solar_PV | 16 | 0.0% | 0.0% | 12.5% | 0.0% | 100.0% | 0.0% |
| Integra_manual | 16 | 0.0% | 6.2% | 62.5% | 6.2% | 100.0% | 6.2% |
| Jungjun_AI_Agent | 16 | 0.0% | 6.2% | 37.5% | 3.1% | 100.0% | 3.1% |
| KI_En_ChatGPT_Praktische_Gids | 16 | 0.0% | 0.0% | 75.0% | 0.0% | 96.9% | 0.0% |
| Kimothi_RAG_Guide | 16 | 0.0% | 0.0% | 75.0% | 6.2% | 87.5% | 6.2% |
| Nagasubramanian_Agentic_AI | 16 | 0.0% | 12.5% | 31.2% | 9.4% | 100.0% | 3.1% |
| PCWorld_July_2025 | 16 | 0.0% | 25.0% | 93.8% | 6.2% | 96.9% | 3.1% |
| Python_Cookbook | 16 | 0.0% | 0.0% | 12.5% | 3.1% | 100.0% | 0.0% |
| Python_Distilled | 16 | 0.0% | 0.0% | 50.0% | 3.1% | 93.8% | 3.1% |
| Raieli_AI_Agents | 16 | 0.0% | 6.2% | 25.0% | 0.0% | 96.9% | 0.0% |
| Recent_Trends_in_Transportation | 16 | 12.5% | 31.2% | 62.5% | 21.9% | 100.0% | 21.9% |
| Sekar_MCP_Standard | 16 | 0.0% | 0.0% | 25.0% | 0.0% | 96.9% | 0.0% |

## 4. Weakest 15 (query, top-1) pairs — v2.10.x defect candidates

- **S0140.Q1** total=0/6 (r=0, f=0, faith=0)
  - Query: 'What are the two types of routing techniques mentioned in the text?'
  - Gold doc: `Kimothi_RAG_Guide` (chunk `12c8aaab4fa1_153_text_a9b113f2`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=312 score=0.505145
  - Judge rationale: The retrieved chunk is completely unrelated to the user query and is not well-formed, making it irrelevant and unhelpful for answering the question about routing techniques.
- **S0014.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the purpose of a greenhouse according to EN-13031-1?'
  - Gold doc: `Greenhouse_Design` (chunk `8b79e9dca3ae_086_text_d8a987c3`)
  - Top-1: `Greenhouse Design and Control by Pedro Ponce.pdf` p=287 score=0.572613
  - Judge rationale: The retrieved chunk does not answer the user query about the purpose of a greenhouse according to EN-13031-1 and contains mostly irrelevant information with minor formatting issues.
- **S0027.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What aircraft did VMFA-232 start using in 1989?'
  - Gold doc: `Combat_Aircraft_August_2025` (chunk `a4c2916a64c2_030_text_3414ebae`)
  - Top-1: `Combat Aircraft - August 2025 UK.pdf` p=29 score=0.533459
  - Judge rationale: The chunk does not answer the query about VMFA-232's aircraft in 1989 and contains minor formatting issues with odd spacing and truncation, making it misleading or wrong for the user's question.
- **S0050.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the main disadvantage of using a dispatch function in Python?'
  - Gold doc: `Fluent_Python` (chunk `1e7e436164a3_229_text_de77be70`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=191 score=0.588468
  - Judge rationale: The retrieved chunk is about a retail knowledge assistant and does not answer the query about the disadvantages of using a dispatch function in Python; it also has minor format issues with odd sentence breaks and context shifts.
- **S0061.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What role does the layer play in reducing model hallucinations?'
  - Gold doc: `Kimothi_RAG_Guide` (chunk `12c8aaab4fa1_179_text_eae9619b`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=191 score=0.610315
  - Judge rationale: The chunk discusses a retail knowledge assistant and GraphRAG, which is not relevant to the role of a layer in reducing model hallucinations, and the information provided is misleading for the user's query.
- **S0066.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What are the key design elements for a minimalist mindfulness e-book cover?'
  - Gold doc: `KI_En_ChatGPT_Praktische_Gids` (chunk `a414e7cb0259_11017_text_0524e699`)
  - Top-1: `A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf` p=64 score=0.54105
  - Judge rationale: The retrieved chunk discusses chunking methods in a technical context and does not address the design elements for a minimalist mindfulness e-book cover, making it irrelevant and unfaithful to the query, with minor formatting issues.
- **S0067.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How does chunking contribute to creating knowledge graphs from documents?'
  - Gold doc: `Kimothi_RAG_Guide` (chunk `12c8aaab4fa1_203_text_611041b5`)
  - Top-1: `A Simple Guide to Retrieval Augmented Generation Kimothi A. 2025.pdf` p=199 score=0.647522
  - Judge rationale: The retrieved chunk does not address the user query about chunking and knowledge graphs, has minor formatting issues, and does not provide a correct or self-contained answer to the question.
- **S0070.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the purpose of the socket module in network programming?'
  - Gold doc: `Python_Distilled` (chunk `24ecec9a39ce_436_text_42e61d4d`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=191 score=0.538867
  - Judge rationale: The retrieved chunk is about a retail knowledge assistant and GraphRAG, which is not related to the purpose of the socket module in network programming, and it contains minor format issues with odd whitespace and truncation.
- **S0076.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How can you create a bytearray instance from a string?'
  - Gold doc: `Python_Distilled` (chunk `24ecec9a39ce_453_text_91b3aea1`)
  - Top-1: `Fluent Python Luciano Ramalho 2015.pdf` p=128 score=0.616076
  - Judge rationale: The retrieved chunk does not answer the query about creating a bytearray from a string and contains unrelated information with minor formatting issues.
- **S0120.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What are the main challenges for commercializing HFCVs?'
  - Gold doc: `Hybrid_electric_vehicles` (chunk `2baf312fdd78_009_text_d7e30e7a`)
  - Top-1: `Hybrid_electric_vehicles_and_their_challenges.pdf` p=12 score=0.619423
  - Judge rationale: The chunk does not address the challenges for commercializing HFCVs and is poorly formatted with fragmented sentences and missing data.
- **S0125.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models use the oliefilter with Mapco, ink.ex.BTW Titel 2,30?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_007_text_7b2e5fc3`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=9 score=0.616269
  - Judge rationale: The chunk does not answer the query about car models using a specific oliefilter and contains minor formatting issues, making it neither relevant nor faithful to the user's question.
- **S0126.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How did the distributed PID controller perform compared to other controllers?'
  - Gold doc: `Greenhouse_Design` (chunk `8b79e9dca3ae_061_text_07583ded`)
  - Top-1: `Greenhouse Design and Control by Pedro Ponce.pdf` p=189 score=0.627775
  - Judge rationale: The retrieved chunk does not answer the user query about the performance of the distributed PID controller compared to other controllers and contains mostly data points without context, making it misleading.
- **S0130.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the purpose of the AIOS-Agent SDK?'
  - Gold doc: `AIOS_LLM_Agent_Operating_System` (chunk `07a1232cccf4_007_text_f9b507ed`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=191 score=0.588699
  - Judge rationale: The retrieved chunk discusses a different topic and does not answer the user query about the AIOS-Agent SDK, and it has minor format issues with odd whitespace and truncation.
- **S0187.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How does HuggingGPT make the results understandable for users?'
  - Gold doc: `Raieli_AI_Agents` (chunk `41b2f4013cff_354_text_70961cd1`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=191 score=0.62313
  - Judge rationale: The retrieved chunk discusses a different topic and does not answer the user query about HuggingGPT, and it has minor formatting issues with odd whitespace and structure.
- **S0205.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What are the main types of server-side vulnerabilities discussed?'
  - Gold doc: `Sekar_MCP_Standard` (chunk `47bcf7e2f91b_186_text_2b523d2a`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=191 score=0.582636
  - Judge rationale: The retrieved chunk discusses a different topic and does not answer the user query about server-side vulnerabilities, it has minor formatting issues, and it is misleading as it provides information unrelated to the question asked.

## 5. Methodology

- Sampled 259 text chunks (≥ 150 chars, ≤ 40% code-like lines, no advertisement keywords). Stratified across the 34-doc canonical corpus.
- Each chunk → 2 queries generated by `qwen-max` (temperature 0.3).
- Each query → top-5 retrieved from `mmrag_v2_8` via Ollama `llava` 4096-dim embeddings.
- Each top-1 chunk → graded by `qwen-max` (temperature 0.0) on relevance / format / faithfulness, each 0-2.
- Gold passage is shown to the judge for context; the judge is instructed NOT to penalize a different-chunk same-document retrieval.

## 6. Revision log

| Date | Change |
|---|---|
| 2026-05-16 | Initial v2.10.0-rc1 soak snapshot. |