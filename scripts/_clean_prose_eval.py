"""Clean retrieval eval for the 3 weak prose books: NATURAL questions (not keyword
bags), MID-document chunks only (skip front/back matter where title/TOC/index pages
are legitimately unretrievable). Settles whether the tail is real. gold@10 + gold@25.
Local GX10 only.
"""
from __future__ import annotations
import json, sys, urllib.request
sys.path.insert(0, "scripts")
from synthetic_soak import _call_vllm
from mmrag_v2.endpoints import endpoint
from mmrag_v2.retrieval import get_reranker, retrieve_hybrid_reranked

DENSE, SPARSE = "mmrag_v3__qwen3_local", "mmrag_v3__bm25_sparse"
WEAK = {"41b2f4013cff":"Raieli","c480fb4e3164":"Bourne","6afeb55a9449":"Jungjun"}
N_PER = 8
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

def gen_question(content):
    msg=[{"role":"user","content":"Read this passage and write ONE natural question a reader "
          "would ask that this passage specifically answers. Write it as a real question, not "
          "keywords. Output only the question.\n\nPASSAGE:\n"+content[:1500]}]
    return (_call_vllm(GX, e.model, msg, temperature=0.0, max_tokens=60) or "").strip().splitlines()[0][:200]

h10=h25=tot=0
for did,label in WEAK.items():
    chunks=[c for c in scroll(did) if len((c.get("content") or "").strip())>=200]
    chunks.sort(key=lambda c:str(c.get("chunk_id")))
    lo,hi=int(len(chunks)*0.10), int(len(chunks)*0.90)  # drop front/back matter
    mid=chunks[lo:hi]
    step=max(1,len(mid)//N_PER); picks=mid[::step][:N_PER]
    a=b=0
    for c in picks:
        gid=c.get("chunk_id"); q=gen_question(c.get("content") or "")
        if not q: continue
        res=retrieve_hybrid_reranked(q,dense_collection=DENSE,sparse_collection=SPARSE,
            top_k_retrieve=50,top_n_return=25,rrf_weights=(1.0,0.25),embed_provider="omlx",reranker=rr)
        ids=[(x.get("payload") or {}).get("chunk_id") for x in res]
        in10=gid in ids[:10]; in25=gid in ids
        a+=in10; b+=in25; h10+=in10; h25+=in25; tot+=1
    print(f"  {label:10s} @10={a}/{len(picks)}  @25={b}/{len(picks)}", flush=True)
print(f"\nCLEAN_PROSE @10={h10}/{tot}={h10/tot:.2f}  @25={h25}/{tot}={h25/tot:.2f}")
