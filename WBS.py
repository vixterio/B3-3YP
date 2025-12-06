import matplotlib.pyplot as plt
import networkx as nx

# ---- YOUR WBS STRUCTURE HERE ----
wbs = {
    "Project": {
        "Hormones": {"Insulin": {"Choice of Insulin": {}, "Risks":{}}, "Glucagon": {"Risks":{}}},
        "Actuators": {"Pumps": {}, "CGMs": {}, "Risks": {}},
        "Models": {
            "Research": {"Research Papers": {}, "Decision Matrix": {}},
            "Development": {"Initial code":{}, "Improvement": {"Exercise Model": {}, "Meal Model": {}}},
            "Validation": {"In-silico Testing": {},}

        }
    }
}
# ---- END STRUCTURE ----


# Build the graph from nested dict with numbering
def add_nodes(G, parent, subtree, parent_num=""):
    child_num = 1
    for name, children in subtree.items():
        if parent_num:
            current_num = f"{parent_num}.{child_num}"
        else:
            current_num = str(child_num)
        
        node_label = f"{current_num}\n{name}"
        G.add_edge(parent, node_label)
        
        if children:
            add_nodes(G, node_label, children, current_num)
        
        child_num += 1


root = list(wbs.keys())[0]
G = nx.DiGraph()
add_nodes(G, root, wbs[root])

# ---- Custom tree layout (NO GRAPHVIZ NEEDED) ----
# replaced hierarchy_pos with a version that allocates horizontal space proportional to subtree size
def get_subtree_size(G, node, memo):
    """Return number of leaf nodes in the subtree of node (used for proportional spacing)."""
    if node in memo:
        return memo[node]
    children = list(G.successors(node))
    if not children:
        memo[node] = 1
    else:
        memo[node] = sum(get_subtree_size(G, c, memo) for c in children)
    return memo[node]

def hierarchy_pos(G, root, width=3.0, vert_gap=0.8, vert_loc=0, xcenter=0.5, memo=None):
    """
    Position nodes in a hierarchy. Child subtrees get horizontal width proportional to their leaf counts,
    which avoids overlapping sibling subtrees.
    """
    if memo is None:
        memo = {}
        # precompute sizes for entire tree
        get_subtree_size(G, root, memo)

    children = list(G.successors(root))
    if not children:
        return {root: (xcenter, vert_loc)}
    total = sum(memo[c] for c in children)
    pos = {}
    left = xcenter - width / 2.0
    cum = left
    for c in children:
        cw = width * (memo[c] / total)  # child width proportional to its subtree size
        cx = cum + cw / 2.0
        pos.update(hierarchy_pos(G, c, width=cw, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=cx, memo=memo))
        cum += cw
    pos[root] = (xcenter, vert_loc)
    return pos

# use a larger overall horizontal width so nodes have plenty of room
pos = hierarchy_pos(G, root, width=4.0, vert_gap=1.0)

# ---- Draw ----
plt.figure(figsize=(16, 12))
nx.draw(
    G,
    pos,
    with_labels=True,
    arrows=False,
    node_size=2400,        # reduced so nodes don't overlap as much
    node_color="skyblue",
    font_size=7,          # slightly larger but should fit now
    font_weight="bold",
)

plt.title("Work Breakdown Structure (Pure Python)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig("wbs.png", dpi=200, bbox_inches='tight')
plt.show()

