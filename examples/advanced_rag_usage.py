#!/usr/bin/env python3
"""
Advanced RAG Usage Example for MM-Converter-V2
==============================================

This example demonstrates TRUE Advanced RAG features:
1. Multimodal embeddings (text + images)
2. Hybrid search (dense semantic + sparse lexical)
3. Cross-modal retrieval (query with text, get images)
4. Hierarchical reranking with breadcrumb boost
5. Asset-aware context building for LLM

Prerequisites:
- Qdrant running: docker run -p 6333:6333 qdrant/qdrant
- Ingestion output: output/ingestion.jsonl + assets/
- LM Studio running with Qwen loaded: http://localhost:1234/v1
"""

import json
import os
import sys
from pathlib import Path

# Add src to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import with proper path handling
try:
    from mmrag_v2.rag import (
        AdvancedRAGPipeline,
        LLMContextFormatter,
        RAGConfig,
        load_ingestion_jsonl,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root with the conda environment activated:")
    print("  conda activate ./env")
    print("  python examples/advanced_rag_usage.py")
    sys.exit(1)

try:
    import requests
except ImportError:
    requests = None


# =============================================================================
# CONFIGURATION
# =============================================================================

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"


def query_lmstudio(messages: list) -> str:
    """Send query to LM Studio's OpenAI-compatible API."""
    if requests is None:
        return "[requests library not installed - install with: pip install requests]"
    
    payload = {
        "model": "local-model",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error querying LM Studio: {e}]"


def example_1_basic_search():
    """
    Example 1: Basic hybrid search demonstrating the difference from Qwen3's pipeline.
    
    Qwen3's pipeline:
    - Only stores image paths as metadata
    - Only dense embeddings (no sparse lexical)
    - No reranking
    
    Our Advanced RAG:
    - Images are ACTUALLY embedded (CLIP vision encoder)
    - Hybrid search (bge-m3 dense + sparse)
    - Cross-encoder reranking
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Hybrid Search")
    print("="*70)
    
    config = RAGConfig(
        collection_name="mmrag_demo",
        enable_reranker=True,
        enable_cross_modal=True,
    )
    
    pipeline = AdvancedRAGPipeline(config)
    pipeline.create_collection(force_recreate=False)
    
    # Load and ingest chunks
    jsonl_path = "output/ingestion.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"❌ File not found: {jsonl_path}")
        print("   Run mmrag-v2 on a document first to generate ingestion.jsonl")
        return
    
    chunks = load_ingestion_jsonl(jsonl_path)
    print(f"📄 Loaded {len(chunks)} chunks from {jsonl_path}")
    
    # Ingest (skip if already done)
    stats = {"text": 0, "image": 0, "table": 0}
    for c in chunks:
        stats[c.modality.value] += 1
    print(f"   Modality breakdown: {stats}")
    
    pipeline.ingest_chunks(chunks)
    
    # Search
    query = "What are the main technical specifications?"
    print(f"\n🔍 Query: '{query}'")
    
    results = pipeline.advanced_retrieval(query, top_k=5)
    
    print("\n📊 Retrieval Results:")
    print("-" * 50)
    
    print(f"\n📝 Text Results ({len(results['text_results'])}):")
    for i, r in enumerate(results['text_results'][:3], 1):
        score = r.get('rerank_score', r.get('score', 0))
        breadcrumbs = r.get('breadcrumb_path', [])
        source = f"{r.get('source_file', 'unknown')} p.{r.get('page_number', '?')}"
        print(f"   {i}. [Score: {score:.3f}] [{source}]")
        if breadcrumbs:
            print(f"      Path: {' > '.join(breadcrumbs[:2])}")
        print(f"      {r.get('content', '')[:100]}...")
    
    print(f"\n🖼️  Cross-Modal Results ({len(results['cross_modal_results'])}):")
    for i, r in enumerate(results['cross_modal_results'][:3], 1):
        print(f"   {i}. [Score: {r.get('score', 0):.3f}]")
        print(f"      Image: {r.get('asset_path', 'N/A')}")
        if r.get('visual_description'):
            print(f"      Description: {r.get('visual_description')[:100]}...")
    
    return results


def example_2_cross_modal_search():
    """
    Example 2: Cross-modal retrieval - the key feature missing from Qwen3's pipeline.
    
    This allows:
    - Text query → Retrieve relevant images
    - Describe what you're looking for, get actual image assets
    - CLIP embeddings align text and vision in shared space
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Cross-Modal Retrieval (Text → Images)")
    print("="*70)
    
    config = RAGConfig(
        collection_name="mmrag_demo",
        enable_cross_modal=True,
    )
    
    pipeline = AdvancedRAGPipeline(config)
    
    # Search for images using text description
    queries = [
        "diagram showing system architecture",
        "chart with performance metrics",
        "photograph of equipment",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Cross-modal search
        results = pipeline.search_cross_modal(query, target_modality="image", top_k=3)
        
        print(f"   Found {len(results)} relevant images:")
        for i, r in enumerate(results, 1):
            print(f"   {i}. Score: {r.get('score', 0):.3f}")
            print(f"      Path: {r.get('asset_path', 'N/A')}")
            if r.get('visual_description'):
                print(f"      VLM: {r.get('visual_description')[:80]}...")


def example_3_full_rag_with_llm():
    """
    Example 3: Complete RAG pipeline with LLM integration.
    
    Shows how to:
    1. Retrieve multimodal context
    2. Format for LLM with image references
    3. Send to local LLM (LM Studio)
    4. Get grounded response with citations
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Full RAG with LLM Integration")
    print("="*70)
    
    config = RAGConfig(
        collection_name="mmrag_demo",
        top_k_rerank=5,
        top_k_images=2,
        enable_reranker=True,
        enable_cross_modal=True,
    )
    
    pipeline = AdvancedRAGPipeline(config)
    
    # Advanced retrieval
    query = "Explain the methodology and show any related diagrams"
    print(f"\n🔍 Query: '{query}'")
    
    results = pipeline.advanced_retrieval(query, top_k=5)
    
    # Build LLM context
    context = pipeline.build_llm_context(query, results, max_tokens=3000)
    
    print(f"\n📋 Context Built:")
    print(f"   - Text chunks: {len(results['text_results'])}")
    print(f"   - Images included: {len(context['images'])}")
    print(f"   - Sources: {len(context['sources'])}")
    
    # Format for LLM
    formatter = LLMContextFormatter()
    
    # Option 1: LM Studio format (text + image descriptions)
    lm_studio_payload = formatter.format_for_lm_studio(context, query)
    
    print("\n💬 Sending to LM Studio...")
    response = query_lmstudio(lm_studio_payload["messages"])
    
    print("\n🤖 LLM Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    # Show sources
    print("\n📚 Sources Used:")
    for src in context['sources']:
        score = src.get('score', 0)
        print(f"   - {src['source_file']} p.{src['page']} (score: {score:.3f})")
    
    # Show images
    if context['images']:
        print("\n🖼️  Images Referenced:")
        for img in context['images']:
            print(f"   - {img['path']}")
            if img.get('description'):
                print(f"     Description: {img['description'][:60]}...")


def example_4_hierarchical_reranking():
    """
    Example 4: Hierarchical reranking using breadcrumb metadata.
    
    Your ingestion schema includes breadcrumb_path which we use to:
    - Boost results with more specific hierarchical context
    - Penalize generic/uncontextualized matches
    - Surface results from relevant document sections
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Hierarchical Re-ranking")
    print("="*70)
    
    config = RAGConfig(
        collection_name="mmrag_demo",
        enable_reranker=True,
    )
    
    pipeline = AdvancedRAGPipeline(config)
    
    query = "implementation details"
    print(f"\n🔍 Query: '{query}'")
    
    # Get candidates
    candidates = pipeline.search(query, top_k=10, filter_modality="text")
    
    print(f"\n📊 Before Reranking (top 5):")
    for i, r in enumerate(candidates[:5], 1):
        breadcrumbs = r.get('breadcrumb_path', [])
        print(f"   {i}. Score: {r.get('score', 0):.3f}")
        print(f"      Breadcrumbs: {' > '.join(breadcrumbs) if breadcrumbs else '(none)'}")
    
    # Apply reranking with hierarchical boost
    reranked = pipeline.rerank(query, candidates)
    
    print(f"\n📊 After Reranking (top 5):")
    for i, r in enumerate(reranked[:5], 1):
        breadcrumbs = r.get('breadcrumb_path', [])
        old_score = r.get('score', 0)
        new_score = r.get('rerank_score', 0)
        print(f"   {i}. Before: {old_score:.3f} → After: {new_score:.3f}")
        print(f"      Breadcrumbs: {' > '.join(breadcrumbs) if breadcrumbs else '(none)'}")
        if breadcrumbs and len(breadcrumbs) > 1:
            print(f"      ↑ Boosted by hierarchy depth: +{min(len(breadcrumbs) * 0.02, 0.1):.2f}")


def compare_with_basic_rag():
    """
    Comparison: Show what Qwen3's pipeline misses vs Advanced RAG.
    """
    print("\n" + "="*70)
    print("COMPARISON: Qwen3 Basic RAG vs MM-Converter Advanced RAG")
    print("="*70)
    
    comparison = """
┌─────────────────────────────┬─────────────────────┬──────────────────────────────┐
│ Feature                     │ Qwen3 Pipeline      │ MM-Converter Advanced RAG    │
├─────────────────────────────┼─────────────────────┼──────────────────────────────┤
│ Image Storage               │ Path as metadata    │ ✓ CLIP vision embeddings     │
│ Image Searchable            │ ✗ No                │ ✓ Cross-modal retrieval      │
│ Hybrid Search               │ ✗ Dense only        │ ✓ Dense + Sparse (bge-m3)    │
│ Sparse/Lexical Retrieval    │ ✗ No                │ ✓ SPLADE-style sparse        │
│ Reranking                   │ ✗ Mentioned only    │ ✓ Cross-encoder (bge-rerank) │
│ Hierarchical Boost          │ ✗ No breadcrumbs    │ ✓ Breadcrumb depth boost     │
│ Visual Descriptions         │ ✗ Not stored        │ ✓ VLM descriptions indexed   │
│ Semantic Context            │ ✗ No prev/next      │ ✓ Prev/next text in payload  │
│ Cross-Modal (Text→Image)    │ ✗ Not possible      │ ✓ CLIP alignment             │
│ Asset-Aware Context         │ ✗ Paths only        │ ✓ Images in LLM context      │
│ Apple Silicon Optimization  │ ✗ Not mentioned     │ ✓ MPS device selection       │
└─────────────────────────────┴─────────────────────┴──────────────────────────────┘

🎯 KEY ADVANTAGES:

1. TRUE Multimodal: Images are embedded and retrievable, not just stored as paths
2. Hybrid Search: Combines semantic (dense) + lexical (sparse) for better recall
3. Cross-Modal: Query with text, get images. Query could be "diagram showing losses"
4. Hierarchical: Uses your breadcrumb metadata to boost relevant sections
5. Semantic Context: Uses prev/next snippets from your IngestionChunk schema
6. Reranking: Cross-encoder reorders initial results for better precision
"""
    print(comparison)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MM-Converter-V2 Advanced RAG Examples")
    print("="*70)
    print("\nThis demonstrates TRUE Advanced RAG vs the basic Qwen3 pipeline.")
    print("\nPrerequisites:")
    print("  1. Qdrant running: docker run -p 6333:6333 qdrant/qdrant")
    print("  2. Processed document: mmrag-v2 process <file.pdf>")
    print("  3. LM Studio running (optional, for example 3)")
    
    # Check prerequisites
    jsonl_path = "output/ingestion.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"\n⚠️  Warning: {jsonl_path} not found.")
        print("   Some examples will show schema explanation only.")
    
    # Run examples
    try:
        if os.path.exists(jsonl_path):
            example_1_basic_search()
            example_2_cross_modal_search()
            example_3_full_rag_with_llm()
            example_4_hierarchical_reranking()
        
        compare_with_basic_rag()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)


if __name__ == "__main__":
    main()
