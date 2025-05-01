import sys
import networkx as nx
import itertools
import math
import time

def load_graph(path):
    G = nx.Graph()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("line"):
                continue
            a, b = line.split()[:2]
            G.add_edge(a, b)
    return G

def brute_force_min_dominating_set(G):
    nodes = list(G.nodes()); n = len(nodes)
    for r in range(1, n+1):
        for combo in itertools.combinations(nodes, r):
            domd = set()
            for u in combo:
                domd |= {u} | set(G.neighbors(u))
            if len(domd) == n:
                return set(combo)
    return None

def branch_and_bound_min_dominating_set(G):
    nodes = list(G.nodes()); n = len(nodes)
    neighs = {u:{u}|set(G.neighbors(u)) for u in nodes}
    Δ = max(len(neighs[u]) for u in nodes)
    best = {'size':n+1, 'set':set()}
    def dfs(dom, domd):
        if len(dom) >= best['size']:
            return
        if len(domd) == n:
            best['size'], best['set'] = len(dom), dom.copy()
            return
        lb = math.ceil((n - len(domd)) / Δ)
        if len(dom) + lb >= best['size']:
            return
        u = next(x for x in nodes if x not in domd)
        dfs(dom|{u}, domd|neighs[u])
        for v in G.neighbors(u):
            if v not in dom:
                dfs(dom|{v}, domd|neighs[v])
    dfs(set(), set())
    return best['set']

def greedy_dominating_set(G):
    dom, domd = set(), set()
    while len(domd) < len(G.nodes()):
        u = max(G.nodes(), key=lambda x: len(({x}|set(G.neighbors(x))) - domd))
        dom.add(u)
        domd |= {u} | set(G.neighbors(u))
    return dom

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Use: python {sys.argv[0]} <dataset.txt>")
        sys.exit(1)

    G = load_graph(sys.argv[1])

    # Menu de choice
    print("Select algorithm(s) to run (\"all\" to run all - NOT RECOMMENDED) (comma separated):")
    print("  1- Brute Force (probably won't compute)")
    print("  2- Branch-and-Bound")
    print("  3- Greedy")
    choice = input("> ").strip().lower()

    choices = set()
    if choice in ("all"):
        choices = {"1", "2", "3"}
    else:
        for token in choice.replace(" ", "").split(","):
            if token in ("1","2","3"):
                choices.add(token)

    if not choices:
        print("No valid option selected. Exiting.")
        sys.exit(1)

    print("\n--- Minimum 1-dominating set ---\n")

    if "1" in choices:
        start = time.perf_counter()
        bf_dom = brute_force_min_dominating_set(G)
        dur = time.perf_counter() - start
        print(f"Brute Force:")
        print(f"  Size = {len(bf_dom)}")
        print(f"  Set  = {bf_dom}")
        print(f"  Time = {dur:.3f} s\n")

    if "2" in choices:
        start = time.perf_counter()
        bnb_dom = branch_and_bound_min_dominating_set(G)
        dur = time.perf_counter() - start
        print(f"Branch-and-Bound:")
        print(f"  Size = {len(bnb_dom)}")
        print(f"  Set  = {bnb_dom}")
        print(f"  Time = {dur:.3f} s\n")

    if "3" in choices:
        start = time.perf_counter()
        greedy_dom = greedy_dominating_set(G)
        dur = time.perf_counter() - start
        print(f"Guloso:")
        print(f"  Size = {len(greedy_dom)}")
        print(f"  Set  = {greedy_dom}")
        print(f"  Time = {dur:.3f} s\n")
