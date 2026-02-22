import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from ml.ingestion import TextIngestion, EmbeddingGenerator
from ml.extraction import LLMExtractor
from ml.graph import KnowledgeGraph
from ml.search import HybridSearchEngine
from ml.intelligence import IntelligenceLayer

def main():
    print("\n" + "="*60)
    print(" NOTE AGENT: DATA TRANSFORMATION PIPELINE VISUALIZATION")
    print("="*60 + "\n")

    # 1. Setup
    print("[INIT] Loading Components...")
    ingestion = TextIngestion()
    
    # Try loading real embedder, else use a mock for visualization
    try:
        embedder = EmbeddingGenerator()
        print("   -> Embedding Model: Loaded (sentence-transformers)")
    except Exception:
        print("   -> Embedding Model: Not found. Using MOCK for visualization.")
        class MockEmbedder:
            def generate_embeddings(self, chunks):
                # Return random 3-dim vectors for demo
                import random
                return [[random.random() for _ in range(3)] for _ in chunks]
            class Model:
                def encode(self, query):
                    import random
                    return [random.random() for _ in range(3)]
            model = Model()
        embedder = MockEmbedder()
        
    extractor = LLMExtractor(verbose=True)
    graph = KnowledgeGraph()
    search_engine = HybridSearchEngine(embedding_generator=embedder, graph=graph)
    intelligence = IntelligenceLayer(graph)

    # 2. Raw Input
    raw_text = """Strategic initiatives for Q1:
1. Launch new product feature by March 15
2. Expand to European market with focus on Germany and France
3. Hire 5 senior engineers to support growth
Key assumptions:
- Budget approved by board
- Engineering capacity available
- Market research completed by January
Open questions:
- What's our go-to-market timeline?
- Do we have regulatory approval for EU expansion?"""
    
    print(f"\n[STEP 1] RAW INPUT TEXT")
    print("-" * 30)
    print(f"'{raw_text}'")
    print("-" * 30)

    # 3. Chunking
    print(f"\n[STEP 2] CHUNKING (Text -> Chunks)")
    chunks = ingestion.chunk_text(raw_text, window_size=20, overlap=5) # Small window to force multiple chunks for demo
    print(f"   -> Strategy: Sliding Window (size=20 tokens, overlap=5)")
    print(f"   -> Result: {len(chunks)} Chunks Generated")
    for i, chunk in enumerate(chunks):
        print(f"      [Chunk {i}] (Tokens: {chunk.token_count}): \"{chunk.text.replace(chr(10), ' ')}\"")

    # 4. Embedding
    print(f"\n[STEP 3] EMBEDDING (Chunks -> Vectors)")
    embeddings = embedder.generate_embeddings(chunks)
    print(f"   -> Result: {len(embeddings)} Vectors Generated")
    for i, vec in enumerate(embeddings):
        # Index for search engine while we are at it
        search_engine.index_chunk(f"chunk-{i}", chunks[i].text, vec)
        # Show snippet
        vec_preview = ", ".join([f"{x:.4f}" for x in vec[:3]])
        print(f"      [Vector {i}] [{vec_preview}, ...]")

    # 5. Structured Extraction (Stage 4)
    print(f"\n[STEP 4] STRUCTURED EXTRACTION — Stage 4 (Text -> Knowledge Objects)")
    print(f"   -> Model: Groq LLM ({extractor.model})")
    extraction_result = extractor.extract(raw_text, note_id="note_demo", chunks=chunks)
    
    # Print per-type counts
    type_counts = {}
    for obj in extraction_result.objects:
        type_counts[obj.type] = type_counts.get(obj.type, 0) + 1
    print(f"   -> Per-type counts: {type_counts}")

    # Objects table (ML_doc.pdf format)
    print(f"\n   ── Objects Table ({len(extraction_result.objects)} rows) ──")
    for obj in extraction_result.objects:
        span_info = f"[{obj.span_start}:{obj.span_end}]" if obj.span_start is not None else ""
        ctx = f" ctx=\"{obj.context}\"" if obj.context else ""
        print(f"      {obj.id} | {obj.type:12s} | \"{obj.canonical_text[:60]}\" | conf={obj.confidence}{ctx} {span_info}")

    # Links table
    print(f"\n   ── Links Table ({len(extraction_result.links)} rows) ──")
    for link in extraction_result.links:
        print(f"      {link.source_id} --[{link.type}]--> {link.target_id} (conf={link.confidence})")

    # Object mentions (provenance)
    print(f"\n   ── Object Mentions ({len(extraction_result.mentions)} rows) ──")
    for m in extraction_result.mentions:
        print(f"      {m.object_id} -> note={m.note_id}, span={m.span_id}, role={m.role}")

    # 6. Knowledge Graph
    print(f"\n[STEP 5] KNOWLEDGE GRAPH CONSTRUCTION (Objects -> Graph)")
    graph.add_objects(extraction_result.objects)
    graph.add_links(extraction_result.links)
    print(f"   -> Graph Stats: {graph.graph.number_of_nodes()} Nodes, {graph.graph.number_of_edges()} Edges")
    print(f"   -> Nodes: {list(graph.graph.nodes())}")

    # 7. Search
    print(f"\n[STEP 6] HYBRID SEARCH (Query -> Ranked Segments)")
    query = "strategic initiatives"
    print(f"   -> Query: '{query}'")
    results = search_engine.search(query)
    for i, res in enumerate(results):
        print(f"      {i+1}. [{res.source.upper()}] Score: {res.score:.4f} | \"{res.text.replace(chr(10), ' ')[:40]}...\"")

    # 8. Intelligence
    print(f"\n[STEP 7] INTELLIGENCE LAYER (Graph -> Insights)")
    intelligence_insights = intelligence.generate_insights()
    for insight in intelligence_insights:
        print(f"   -> [INSIGHT] {insight['type']}: {insight['text']} ({insight['message']})")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
