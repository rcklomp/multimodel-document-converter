"""Why do Raieli/Bourne/Jungjun miss? For each real-miss, show the GOLD chunk we tried
to retrieve (is it garbled/empty -> extraction defect?) and the top-3 that came back
instead (from which book, junk or off-topic -> embedding/collision). Local only.
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

for did,label in WEAK.items():
    chunks=[c for c in scroll(did) if len((c.get("content") or "").strip())>=150]
    chunks.sort(key=lambda c:str(c.get("chunk_id")))
    step=max(1,len(chunks)//N_PER); picks=chunks[::step][:N_PER]
    print(f"\n===== {label} (total eligible chunks={len(chunks)}) =====")
    for c in picks:
        gid=c.get("chunk_id"); q=gen_query(c.get("content") or "")
        res=retrieve_hybrid_reranked(q,dense_collection=DENSE,sparse_collection=SPARSE,
            top_k_retrieve=50,top_n_return=10,rrf_weights=(1.0,0.25),embed_provider="omlx",reranker=rr)
        ids=[(x.get("payload") or {}).get("chunk_id") for x in res]
        if gid in ids: continue  # only inspect misses
        print(f"\n MISS q={q[:80]!r}")
        print(f"  GOLD[{gid}]: {(c.get('content') or '')[:180]!r}")
        for i,x in enumerate(res[:3]):
            pl=x.get("payload") or {}
            print(f"  top{i+1} doc={str(pl.get('chunk_id'))[:12]} : {(pl.get('content') or '')[:120]!r}")
