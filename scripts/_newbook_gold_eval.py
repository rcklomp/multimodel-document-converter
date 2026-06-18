"""Full-corpus gold-chunk@10 for the 14 NEWLY-ingested books (bypasses synthetic_soak,
whose doc-name mapping doesn't match the omlx prod collection). Samples gold chunks
DIRECTLY from production, generates one retrieval query per chunk via the GX10 local
LLM (no cloud, no Claude budget), and checks the gold chunk lands in top-10 on the
production hybrid path. Combine with the prior-29-doc 85.8% for the full picture.
"""
from __future__ import annotations
import json, re, sys, urllib.request
sys.path.insert(0, "scripts")
from synthetic_soak import _call_vllm
from mmrag_v2.endpoints import endpoint
from mmrag_v2.retrieval import get_reranker, retrieve_hybrid_reranked

DENSE, SPARSE = "mmrag_v3__qwen3_local", "mmrag_v3__bm25_sparse"
BOOKS = {  # doc_id -> label
 "5b915c809145":"Devlin","d57cdf0231fa":"Zephyr(C)","131b7b54c411":"Adedeji","c480fb4e3164":"Bourne",
 "57dd24faf967":"Chaubal","1e7e436164a3":"FluentPython","70930ff6f3a8":"Hao","6afeb55a9449":"Jungjun",
 "8e184dc5d9c4":"Nagasub","5917d010344e":"ArcGIS","0326dff0bbb4":"PyCookbook","24ecec9a39ce":"PyDistilled",
 "41b2f4013cff":"Raieli","2f9c26fcf758":"CppManual"}
N_PER = 5

e = endpoint("chat"); GX = getattr(e, "chat_url", None) or e.base_url
rr = get_reranker("omlx")

def scroll(doc_id):
    out, off = [], None
    while True:
        b={"limit":1000,"with_payload":["content","chunk_id","modality","chunk_type"],"with_vector":False,
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

tot=hit=0; per={}
for did,label in BOOKS.items():
    chunks=[c for c in scroll(did) if len((c.get("content") or "").strip())>=150]
    chunks.sort(key=lambda c:str(c.get("chunk_id")))
    if not chunks: per[label]="no eligible chunks"; continue
    step=max(1,len(chunks)//N_PER); picks=chunks[::step][:N_PER]
    h=t=0
    for c in picks:
        gid=c.get("chunk_id"); q=gen_query(c.get("content") or "")
        if not q: continue
        try:
            res=retrieve_hybrid_reranked(q,dense_collection=DENSE,sparse_collection=SPARSE,
                top_k_retrieve=50,top_n_return=10,rrf_weights=(1.0,0.25),embed_provider="omlx",reranker=rr)
        except Exception as ex: print(f"! {label} {gid}: {ex}"); continue
        ids=[(x.get("payload") or {}).get("chunk_id") for x in res]
        ok=gid in ids; h+=ok; t+=1; hit+=ok; tot+=1
    per[label]=f"{h}/{t}"
    print(f"  {label:14s} {h}/{t}", flush=True)
print(f"\nNEWBOOK_GOLD@10: {hit}/{tot} = {hit/tot:.3f}" if tot else "no queries")
