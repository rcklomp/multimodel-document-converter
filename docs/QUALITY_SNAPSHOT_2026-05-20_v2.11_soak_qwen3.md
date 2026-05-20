# Quality Snapshot 2026-05-20 — SOAK (mmrag_v2_8__qwen3_dashscope)

> **Status:** synthetic-soak report.
> Source: `output/soak/v2.11_qwen3/work.jsonl`.
> Judge: Dashscope `qwen-max`. Generator: `qwen-max`. Embedder: `text-embedding-v4` (provider=dashscope). Collection: `mmrag_v2_8__qwen3_dashscope`.
> No QA threshold; this snapshot is informational.

## 1. Corpus summary

- Sampled chunks: **259** across 33 docs.
- Queries generated: **518**.
- Queries judged: **518/518** (100.0%).

## 2. Headline metrics

| Metric | Value |
|---|---:|
| Recall@1 (gold chunk_id is top-1) | 35.5% (184/518) |
| Recall@5 (gold chunk_id in top-5) | 66.8% (346/518) |
| Recall@5 (gold doc_id in top-5)   | 91.7% (475/518) |
| Relevance score                   | 59.3% (614/1036) |
| Format score                      | 89.8% (930/1036) |
| Faithfulness score                | 50.6% (524/1036) |

## 3. Per-document metrics

| Doc | Queries | R@1 | R@5 (chunk) | R@5 (doc) | Relevance | Format | Faith |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIOS_LLM_Agent_Operating_System | 16 | 50.0% | 75.0% | 87.5% | 65.6% | 93.8% | 59.4% |
| ATZ_Elektronik_German | 16 | 62.5% | 81.2% | 93.8% | 59.4% | 87.5% | 46.9% |
| A_comprehensive_review_on_hybrid_electri | 16 | 18.8% | 68.8% | 100.0% | 56.2% | 78.1% | 46.9% |
| Adedeji_GenAI_Google_Cloud | 16 | 43.8% | 75.0% | 100.0% | 59.4% | 93.8% | 53.1% |
| ArcGIS_Python_Cookbook | 16 | 12.5% | 50.0% | 81.2% | 46.9% | 93.8% | 37.5% |
| Ayeva_Python_Patterns | 16 | 31.2% | 81.2% | 93.8% | 50.0% | 93.8% | 43.8% |
| Bourne_RAG_2024 | 16 | 18.8% | 50.0% | 81.2% | 62.5% | 96.9% | 43.8% |
| CarOK_voorraadtelling | 16 | 12.5% | 43.8% | 100.0% | 40.6% | 68.8% | 18.8% |
| ChatGPT_Praktijk_handboek | 16 | 18.8% | 56.2% | 81.2% | 43.8% | 100.0% | 37.5% |
| Chaubal_PyTorch_Projects | 16 | 31.2% | 56.2% | 87.5% | 62.5% | 96.9% | 56.2% |
| Combat_Aircraft_August_2025 | 16 | 50.0% | 93.8% | 100.0% | 62.5% | 84.4% | 62.5% |
| Cronin_GenAI_Models | 16 | 31.2% | 56.2% | 68.8% | 62.5% | 100.0% | 59.4% |
| Devlin_LLM_Agents | 16 | 43.8% | 62.5% | 75.0% | 59.4% | 78.1% | 50.0% |
| Earthship_Vol1 | 16 | 50.0% | 75.0% | 100.0% | 62.5% | 71.9% | 53.1% |
| Firearms | 16 | 31.2% | 75.0% | 100.0% | 46.9% | 84.4% | 37.5% |
| Fluent_Python | 16 | 50.0% | 81.2% | 93.8% | 81.2% | 100.0% | 75.0% |
| Form_betwistingsformulier | 6 | 50.0% | 66.7% | 83.3% | 50.0% | 100.0% | 50.0% |
| Greenhouse_Design | 16 | 25.0% | 87.5% | 100.0% | 75.0% | 84.4% | 59.4% |
| Hao_ML_Platform | 16 | 31.2% | 62.5% | 100.0% | 56.2% | 93.8% | 50.0% |
| HarryPotter_and_the_Sorcerers_Stone | 16 | 62.5% | 81.2% | 100.0% | 75.0% | 84.4% | 68.8% |
| Hybrid_electric_vehicles | 16 | 68.8% | 81.2% | 100.0% | 62.5% | 90.6% | 59.4% |
| IRJET_Modeling_of_Solar_PV | 16 | 50.0% | 75.0% | 93.8% | 71.9% | 71.9% | 53.1% |
| Integra_manual | 16 | 37.5% | 62.5% | 100.0% | 50.0% | 93.8% | 43.8% |
| Jungjun_AI_Agent | 16 | 12.5% | 43.8% | 81.2% | 40.6% | 96.9% | 34.4% |
| KI_En_ChatGPT_Praktische_Gids | 16 | 25.0% | 50.0% | 100.0% | 56.2% | 100.0% | 40.6% |
| Kimothi_RAG_Guide | 16 | 12.5% | 43.8% | 68.8% | 56.2% | 96.9% | 37.5% |
| Nagasubramanian_Agentic_AI | 16 | 25.0% | 37.5% | 87.5% | 46.9% | 87.5% | 43.8% |
| PCWorld_July_2025 | 16 | 37.5% | 81.2% | 100.0% | 84.4% | 96.9% | 81.2% |
| Python_Cookbook | 16 | 31.2% | 62.5% | 75.0% | 65.6% | 87.5% | 65.6% |
| Python_Distilled | 16 | 43.8% | 81.2% | 100.0% | 62.5% | 87.5% | 53.1% |
| Raieli_AI_Agents | 16 | 56.2% | 75.0% | 93.8% | 75.0% | 90.6% | 65.6% |
| Recent_Trends_in_Transportation | 16 | 31.2% | 81.2% | 93.8% | 46.9% | 87.5% | 40.6% |
| Sekar_MCP_Standard | 16 | 25.0% | 50.0% | 100.0% | 53.1% | 96.9% | 40.6% |

## 4. Weakest 15 (query, top-1) pairs — v2.10.x defect candidates

- **S0159.Q2** total=0/6 (r=0, f=0, faith=0)
  - Query: 'How should the process be handled if major flaws are identified in the answer?'
  - Gold doc: `Nagasubramanian_Agentic_AI` (chunk `8e184dc5d9c4_328_text_0c36c2a9`)
  - Top-1: `Raieli S. Building AI Agents with LLMs, RAG, and Knowledge Graphs...2025.pdf` p=118 score=0.485034
  - Judge rationale: The retrieved chunk describes a graphic layout and does not address the process for handling major flaws in an answer, nor is it in a readable prose format.
- **S0018.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How can bridging internal silos benefit an organization?'
  - Gold doc: `Devlin_LLM_Agents` (chunk `5b915c809145_185_text_f8cf71aa`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=185 score=0.390646
  - Judge rationale: The chunk does not provide an answer to the query about the benefits of bridging internal silos and has repeated words affecting readability and clarity.
- **S0023.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How does the server reveal its user ID and why is this a security risk?'
  - Gold doc: `Sekar_MCP_Standard` (chunk `47bcf7e2f91b_202_text_662416ae`)
  - Top-1: `Sekar S. The MCP Standard. A Developer's Guide..Building Universal AI Tools 2026.pdf` p=113 score=0.547913
  - Judge rationale: The retrieved chunk does not address the user query about server revealing user ID and its security risk; it discusses unrelated system tools and notifications with some formatting issues.
- **S0039.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: "How does the PER agent process a user's query about AI research and leaders?"
  - Gold doc: `Nagasubramanian_Agentic_AI` (chunk `8e184dc5d9c4_145_text_afa95f4a`)
  - Top-1: `Devlin M. Building LLM Agents with RAG, Knowledge Graphs and Reflection...2025.pdf` p=214 score=0.600131
  - Judge rationale: The chunk discusses a different query and system process, not answering the user's query about AI research and leaders; it has minor format issues with repetition and is misleading as it does not match the user's request.
- **S0044.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How does the F-35A Lightning II improve upon the capabilities of the F-15 Eagle?'
  - Gold doc: `Combat_Aircraft_August_2025` (chunk `a4c2916a64c2_013_text_f06341e9`)
  - Top-1: `Combat Aircraft - August 2025 UK.pdf` p=25 score=0.585661
  - Judge rationale: The retrieved chunk discusses the relocation of F-15E units and does not address how the F-35A improves upon the F-15 Eagle; it also contains truncation and minor formatting issues.
- **S0071.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How does voltage balancing control benefit MPPT systems?'
  - Gold doc: `A_comprehensive_review_on_hybrid_electri` (chunk `1b6ba953d1f4_018_text_1d32b576`)
  - Top-1: `IRJET_Modeling_of_Solar_PV_system_under.pdf` p=5 score=0.592762
  - Judge rationale: The retrieved chunk does not address the benefits of voltage balancing control for MPPT systems and contains minor formatting issues with odd whitespace and truncation, making it both irrelevant and not self-contained for answering the query correctly.
- **S0093.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How do you install Faker and ReactiveX for the project?'
  - Gold doc: `Ayeva_Python_Patterns` (chunk `289fd158f828_206_text_df907096`)
  - Top-1: `Ayeva K. Mastering Python Design Patterns...essential Python patterns...3ed 2024.pdf` p=221 score=0.620901
  - Judge rationale: The chunk discusses an Observable and does not provide installation instructions for Faker and ReactiveX, and the format is slightly off due to truncation.
- **S0098.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which car models are compatible with the oliefilter having art_nr_merk 61551?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_008_text_6eb2d7fc`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=3 score=0.684523
  - Judge rationale: The retrieved chunk does not contain information about the car models compatible with art_nr_merk 61551 and is not from the same domain, it also has minor formatting issues but is mostly readable, and provides no correct or relevant answer to the user's query.
- **S0109.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How can you view the contents of a ConfigMap using kubectl?'
  - Gold doc: `Hao_ML_Platform` (chunk `70930ff6f3a8_092_text_d31386ff`)
  - Top-1: `Hao B. Machine Learning Platform Engineering. Build...for ML and AI systems 2026.pdf` p=209 score=0.623462
  - Judge rationale: The retrieved chunk does not answer the user query about viewing ConfigMap contents with kubectl and is unrelated to the topic, though it is marginally well-formed.
- **S0120.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'How does a Plug-In HEV recharge its battery?'
  - Gold doc: `Hybrid_electric_vehicles` (chunk `2baf312fdd78_009_text_d7e30e7a`)
  - Top-1: `Hybrid_electric_vehicles_and_their_challenges.pdf` p=9 score=0.596531
  - Judge rationale: The retrieved chunk does not answer the query about how a Plug-In HEV recharges its battery and contains some minor formatting issues with incomplete sentences and formula placeholders.
- **S0129.Q2** total=1/6 (r=1, f=0, faith=0)
  - Query: 'How have power converter topologies impacted the traction systems for electric vehicles?'
  - Gold doc: `A_comprehensive_review_on_hybrid_electri` (chunk `1b6ba953d1f4_013_text_e0a4325d`)
  - Top-1: `A_comprehensive_review_on_hybrid_electri.pdf` p=24 score=0.615827
  - Judge rationale: The chunk is about related topics but does not directly answer the query, and it is poorly formatted with severe truncation, making it difficult to understand and potentially misleading.
- **S0144.Q2** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Why is it not advisable to expand a single room to make a house?'
  - Gold doc: `Earthship_Vol1` (chunk `aa11d5ea2275_055_text_fc06da69`)
  - Top-1: `Earthship_Vol1_How to build your own.pdf` p=59 score=0.529017
  - Judge rationale: The retrieved chunk discusses the placement of rooms for heating purposes but does not address why a single room should not be expanded to make a house, and it has minor formatting issues with odd spacing and truncation.
- **S0159.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'What is the purpose of treating the critique as a separate internal reasoning step?'
  - Gold doc: `Nagasubramanian_Agentic_AI` (chunk `8e184dc5d9c4_328_text_0c36c2a9`)
  - Top-1: `Nagasubramanian D. Agentic AI for Engineers.Architecting Goal-Driven System 2026.pdf` p=317 score=0.568614
  - Judge rationale: The retrieved chunk does not address the purpose of treating the critique as a separate step and is mostly code with minor formatting issues, making it irrelevant and unhelpful for the user query.
- **S0171.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Why is it recommended to use a dedicated main() function in Python scripts?'
  - Gold doc: `Python_Distilled` (chunk `24ecec9a39ce_389_text_3ed68439`)
  - Top-1: `Python Distilled David M. Beazley 2022.pdf` p=973 score=0.642441
  - Judge rationale: The retrieved chunk does not address the query about using a dedicated main() function and is only a code snippet without context, making it neither relevant nor self-contained for the user's question.
- **S0198.Q1** total=1/6 (r=0, f=1, faith=0)
  - Query: 'Which vehicles are compatible with the Mapco oil filter 61098?'
  - Gold doc: `CarOK_voorraadtelling` (chunk `46d689134b24_007_text_672f1c0b`)
  - Top-1: `CarOK voorraadtelling 2021-04.pdf` p=8 score=0.688659
  - Judge rationale: The retrieved chunk does not mention the Mapco oil filter 61098 or any compatible vehicles, and it has minor formatting issues with odd truncation at the end.

## 5. Methodology

- Sampled 259 text chunks (≥ 150 chars, ≤ 40% code-like lines, no advertisement keywords). Stratified across the 34-doc canonical corpus.
- Each chunk → 2 queries generated by `qwen-max` (temperature 0.3).
- Each query → top-5 retrieved from `mmrag_v2_8__qwen3_dashscope` via `dashscope` provider, model `text-embedding-v4`.
- Each top-1 chunk → graded by `qwen-max` (temperature 0.0) on relevance / format / faithfulness, each 0-2.
- Gold passage is shown to the judge for context; the judge is instructed NOT to penalize a different-chunk same-document retrieval.

## 6. Revision log

| Date | Change |
|---|---|
| 2026-05-20 | Initial v2.10.0-rc1 soak snapshot. |