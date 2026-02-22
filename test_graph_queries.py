import networkx as nx
from ml.graph import KnowledgeGraph
from ml.extraction import ExtractedObject, Link

def test_graph_intelligence():
    print("\n[GRAPH TEST] INITIALIZING KNOWLEDGE GRAPH...")
    kg = KnowledgeGraph()

    # 1. Mock Data from Demo Output
    objects = [
        ExtractedObject(id="obj_001", type="Idea", canonical_text="Strategic initiatives for Q1", confidence=1.0),
        ExtractedObject(id="obj_004", type="Task", canonical_text="Launch product feature", confidence=1.0),
        ExtractedObject(id="obj_007", type="Assumption", canonical_text="Budget approved", confidence=1.0),
        ExtractedObject(id="obj_006", type="Task", canonical_text="Hire 5 engineers", confidence=1.0),
        ExtractedObject(id="obj_008", type="Assumption", canonical_text="Engineering capacity available", confidence=1.0),
    ]
    
    links = [
        Link(source_id="obj_004", target_id="obj_007", type="DependsOn", confidence=0.9),
        Link(source_id="obj_006", target_id="obj_007", type="DependsOn", confidence=0.8),
        Link(source_id="obj_006", target_id="obj_008", type="Contradicts", confidence=0.7), # A test contradiction
        Link(source_id="obj_001", target_id="obj_004", type="Supports", confidence=0.9),
    ]

    kg.add_objects(objects)
    kg.add_links(links)

    # 2. Test Centrality (The "Core Concepts")
    print("\n[QUERY 1] IDENTIFYING CORE CONCEPTS (Centrality)...")
    centrality = kg.custom_centrality()
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    for node_id, score in sorted_centrality[:3]:
        text = kg.graph.nodes[node_id]['canonical_text']
        print(f"   -> {node_id}: \"{text}\" (Centrality: {score:.2f})")

    # 3. Test Contradictions
    print("\n[QUERY 2] FINDING SEMANTIC CONTRADICTIONS...")
    contradictions = kg.find_contradictions()
    if contradictions:
        for c in contradictions:
            print(f"   ⚠️ WARNING: \"{c['source']['canonical_text']}\" CONTRADICTS \"{c['target']['canonical_text']}\" (Conf: {c['edge']['confidence']})")
    else:
        print("   ✓ No contradictions found.")

    # 4. Test Subgraph (Context)
    print("\n[QUERY 3] RETRIEVING SUBGRAPH FOR 'obj_006' (Context)...")
    subgraph = kg.get_subgraph("obj_006", depth=1)
    print(f"   -> Found {len(subgraph['nodes'])} related nodes and {len(subgraph['links'])} connections.")
    for link in subgraph['links']:
        print(f"      - {link['source']} --[{link['type']}]--> {link['target']}")

if __name__ == "__main__":
    test_graph_intelligence()
