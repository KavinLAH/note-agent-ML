"""
Stage 4: Knowledge Graph Visualization
Executive-presentation-ready: large readable nodes, full text, clear arrows.

Usage:
    python visualize_graph.py

Output:
    knowledge_graph.png in the project root
"""
import os

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import textwrap
import networkx as nx

from ml.extraction import LLMExtractor
from ml.graph import KnowledgeGraph


# ── Colors ──
TYPE_COLORS = {
    'Idea':       '#3B82F6',
    'Claim':      '#22C55E',
    'Assumption': '#EAB308',
    'Question':   '#8B5CF6',
    'Task':       '#EC4899',
    'Evidence':   '#06B6D4',
    'Definition': '#6366F1',
}

EDGE_COLORS = {
    'DependsOn':   '#FBBF24',
    'Supports':    '#60A5FA',
    'Causes':      '#FB923C',
    'RefersTo':    '#A78BFA',
    'Contradicts': '#F87171',
    'Refines':     '#34D399',
    'SameAs':      '#94A3B8',
}


def visualize(graph: KnowledgeGraph, output_path: str = 'knowledge_graph.png'):
    """Executive-ready knowledge graph with auto-sized boxes."""
    G = graph.graph
    if len(G.nodes()) == 0:
        print("[Visualize] No nodes to draw.")
        return

    fig, ax = plt.subplots(figsize=(22, 14))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')

    pos = _presentation_layout(G)

    # Set axis limits FIRST so data coordinates are stable
    xvals = [p[0] for p in pos.values()]
    yvals = [p[1] for p in pos.values()]
    ax.set_xlim(min(xvals) - 4, max(xvals) + 4)
    ax.set_ylim(min(yvals) - 2, max(yvals) + 2)

    # ── Draw nodes using matplotlib's bbox (auto-sized to text) ──
    node_positions = {}
    for node in G.nodes():
        data = G.nodes[node]
        obj_type = data.get('type', 'Unknown')
        raw_text = data.get('canonical_text', node)
        color = TYPE_COLORS.get(obj_type, '#64748B')
        x, y = pos[node]

        # Wrap text and add type header - Show full text for research transparency
        wrapped = textwrap.fill(raw_text, width=25)
        label = f"[{obj_type.upper()}]\n{wrapped}"

        # Use matplotlib's bbox — it auto-sizes perfectly to text
        ax.text(x, y, label,
            fontsize=13, fontweight='bold', color='#FFFFFF',
            ha='center', va='center', fontfamily='sans-serif',
            linespacing=1.4, zorder=10,
            bbox=dict(
                boxstyle='round,pad=0.6',
                facecolor=color,
                edgecolor='#FFFFFF',
                linewidth=2.0,
                alpha=0.9, # Glassmorphism/translucency restored
            ))

        node_positions[node] = (x, y)

    # ── Draw edges (fan out duplicates between same node pair) ──
    pair_count = {}  # track how many edges between each pair
    for u, v, data in G.edges(data=True):
        pair = (min(u, v), max(u, v))
        idx = pair_count.get(pair, 0)
        pair_count[pair] = idx + 1

        lt = data.get('type', 'Supports')
        ec = EDGE_COLORS.get(lt, '#94A3B8')
        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]

        # Fan out: wide alternating curvature for duplicate edges
        if idx == 0:
            rad = 0.0
        else:
            sign = 1 if idx % 2 == 1 else -1
            # Increased base curvature for better separation of duplicate lines
            rad = sign * (0.45 + 0.25 * ((idx - 1) // 2))

        # Cross-column edges: gentle curve to avoid crossing nodes
        col_dist = abs(x1 - x2)
        row_dist = abs(y1 - y2)
        if col_dist > 1 and row_dist > 1 and rad == 0:
            rad = 0.25 if col_dist > 8 else 0.15

        # Arrow with subtle glow effect restored
        # Glow layer (wider, translucent)
        ax.annotate('',
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='-', color=ec, lw=6, alpha=0.15,
                connectionstyle=f'arc3,rad={rad}',
                shrinkA=45, shrinkB=45,
            ),
            zorder=4)

        # Main arrow - increased mutation_scale for massive arrowheads
        ax.annotate('',
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='-|>', color=ec, lw=2.2, 
                mutation_scale=30, 
                connectionstyle=f'arc3,rad={rad}',
                shrinkA=45, 
                shrinkB=50, 
                alpha=0.9, 
            ),
            zorder=5)
        
        # ── Edge label ON the Bézier curve with background pill ──
        midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        cx = midx + rad * dy
        cy = midy - rad * dx
        
        # t=0.5 is the exact center of the curve.
        # For long executive-spaced lines, staying centered is clearest.
        t = 0.5
            
        bx = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
        by = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2

        # Small nudge to place label ABOVE the curve rather than ON it
        length = max((dx**2 + dy**2) ** 0.5, 0.001)
        px, py = -dy / length, dx / length
        nudge_dist = 0.5  # Fixed distance from the line
        
        # Ensure nudge is always "outward" from the center of curvature
        nudge_sign = 1 if rad >= 0 else -1
        bx += px * nudge_sign * nudge_dist
        by += py * nudge_sign * nudge_dist

        # Background pill + text
        ax.text(bx, by, f'  {lt}  ',
            fontsize=10, color='#FFFFFF', fontweight='bold',
            ha='center', va='center', fontfamily='sans-serif',
            zorder=12,
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor=ec, edgecolor='#FFFFFF',
                linewidth=1.0,
                alpha=1.0, 
            ))

    # ── Title ──
    ax.text(0.5, 0.97, 'Knowledge Graph  —  Stage 4 Extraction',
        transform=ax.transAxes, fontsize=24, fontweight='bold',
        color='#F8FAFC', ha='center', va='top', fontfamily='sans-serif')
    ax.text(0.5, 0.935, 'Objects & Relationships  •  Llama 3.3 70B via Groq',
        transform=ax.transAxes, fontsize=13, color='#64748B',
        ha='center', va='top', fontfamily='sans-serif')

    # ── Legend ──
    legend_types = sorted(set(G.nodes[n].get('type', '?') for n in G.nodes()))
    for i, t in enumerate(legend_types):
        c = TYPE_COLORS.get(t, '#64748B')
        ax.text(0.02 + i * 0.12, 0.04, f"● {t}",
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=c, ha='left', va='bottom', fontfamily='sans-serif')

    ax.axis('off')
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'[Visualize] Saved: {output_path}')
    plt.close()


def _presentation_layout(G):
    """Roots at top, chains flowing downward, wide spacing."""
    in_deg = dict(G.in_degree())
    roots = [n for n, d in in_deg.items() if d == 0]
    if not roots:
        roots = list(G.nodes())[:1]

    chains = []
    assigned = set()

    for root in roots:
        chain = []
        current = root
        while current and current not in assigned:
            chain.append(current)
            assigned.add(current)
            nexts = [s for s in G.successors(current) if s not in assigned]
            current = nexts[0] if nexts else None
        if chain:
            chains.append(chain)

    remaining = [n for n in G.nodes() if n not in assigned]
    if remaining:
        chains.append(remaining)

    pos = {}
    col_spacing = 7.5
    row_spacing = 5.0
    total_width = (len(chains) - 1) * col_spacing
    start_x = -total_width / 2

    for ci, chain in enumerate(chains):
        col_x = start_x + ci * col_spacing
        for ri, node in enumerate(chain):
            pos[node] = (col_x, -ri * row_spacing)

    return pos


if __name__ == '__main__':
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

    extractor = LLMExtractor()
    result = extractor.extract(raw_text, note_id='note_demo', span_id='span_full')

    graph = KnowledgeGraph()
    graph.add_objects(result.objects)
    graph.add_links(result.links)

    output = os.path.join(os.path.dirname(__file__), 'knowledge_graph.png')
    visualize(graph, output)

    import platform
    if platform.system() == 'Darwin':
        os.system(f'open {output}')
