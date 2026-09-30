"""Tail diagnostic: for the weak prose books, distinguish a REAL retrieval miss from
gold-chunk-recall pessimism. Gold-chunk@10 demands the EXACT sampled chunk; in a corpus
with many books on the same topic (AI agents / RAG) a query's relevant answer may be a
different, equally-valid chunk. For each gold chunk in the weak books we regenerate the
same query (temp 0), retrieve top-10, and when the gold chunk is ABSENT we ask the GX10
judge whether any of the top-3 results actually ANSWERS the query. real_miss = gold
absent AND no relevant top-3. Local GX10 only.
"""
from __future__ import annotations
import json, sys, urllib.request
sys.path.insert(0, "scripts")
from synthetic_soak import _call_vllm
from mmrag_v2.endpoints import endpoint
from mmrag_v2.retrieval import get_reranker, retrieve_hybrid_reranked

DENSE, SPARSE = "mmrag_v3__qwen3_local", "mmrag_v3__bm25_sparse"
WEAK = {"41b2f4013cff":"Raieli","c480fb4e3164":"Bourne","6afeb55a9449":"Jungjun"}
N_PER = 5
e = endpoint("chat"); GX = getattr(e, "chat_url", None) or e.base_url
rr = get_reranker("omlx")

def scroll(doc_id):
    out, off = [], None
    while True:
        b={"limit":1000,"with_payload":["content","chunk_id"],"with_vector":False,
           "filter":{"must":[{"key":"doc_id","match":{"value":doc_id}}]}}
        if off: b["offset"]=off
        r=json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://localhost:6333/collections/{DENSE}/points/scroll",
            data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})).read())["result"]
        out+=[p.get("payload") or {} for p in r["points"]]; off=r.get("next_page_offset")
        if not off: break
    return out

def gen_query(content):
    msg=[{"role":"user","content":"Write ONE specific search query (8-16 words) a developer "
          "would type to find this exact passage. Output only the query, no quotes.\n\nPASSAGE:\n"+content[:1500]}]
    return (_call_vllm(GX, e.model, msg, temperature=0.0, max_tokens=60) or "").strip().splitlines()[0][:200]

def judge_relevant(query, passages):
    body=("Query: "+query+"\n\nCandidate passages:\n"
          +"\n---\n".join(f"[{i+1}] {p[:600]}" for i,p in enumerate(passages))
          +"\n\nDoes ANY candidate substantively answer the query? Reply only YES or NO.")
    a=(_call_vllm(GX, e.model, [{"role":"user","content":body}], temperature=0.0, max_tokens=5) or "").upper()
    return "YES" in a

real_miss=pessimism=gold_hit=0
for did,label in WEAK.items():
    chunks=[c for c in scroll(did) if len((c.get("content") or "").strip())>=150]
    chunks.sort(key=lambda c:str(c.get("chunk_id")))
    step=max(1,len(chunks)//N_PER); picks=chunks[::step][:N_PER]
    for c in picks:
        gid=c.get("chunk_id"); q=gen_query(c.get("content") or "")
        if not q: continue
        res=retrieve_hybrid_reranked(q,dense_collection=DENSE,sparse_collection=SPARSE,
            top_k_retrieve=50,top_n_return=10,rrf_weights=(1.0,0.25),embed_provider="omlx",reranker=rr)
        ids=[(x.get("payload") or {}).get("chunk_id") for x in res]
        if gid in ids:
            gold_hit+=1; continue
        top3=[(x.get("payload") or {}).get("content") or "" for x in res[:3]]
        if judge_relevant(q, top3):
            pessimism+=1; print(f"  {label}: gold absent BUT top-3 relevant (pessimism)", flush=True)
        else:
            real_miss+=1; print(f"  {label}: REAL MISS - q={q[:70]!r}", flush=True)
print(f"\nTAIL: gold_hit={gold_hit} pessimism(relevant-but-not-gold)={pessimism} real_miss={real_miss}")
