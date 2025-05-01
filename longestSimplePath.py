import sys
import networkx as nx
import itertools
from collections import deque
import math
import random
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

def brute_force_longest_cycle(G, start):
    best_path = []
    def dfs(path, visited):
        curr = path[-1]
        for w in G.neighbors(curr):
            if w == start and len(path) > 1:
                if len(path) > len(best_path):
                    best_path[:] = path
            elif w not in visited:
                visited.add(w); path.append(w)
                dfs(path, visited)
                path.pop(); visited.remove(w)
    dfs([start], {start})
    return best_path

def branch_and_bound_longest_cycle(G, start):
    best = {'length':0, 'path':[]}
    adj = {u:list(G.neighbors(u)) for u in G.nodes()}
    def reachable_count(u, vis):
        seen = set(vis); q = deque([u]); seen.add(u); cnt = 1
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y); q.append(y); cnt += 1
        return cnt
    def dfs(path):
        curr = path[-1]; L = len(path)
        if L + reachable_count(curr, path) - 1 <= best['length']:
            return
        for w in adj[curr]:
            if w == start and L > 1:
                if L > best['length']:
                    best['length'], best['path'] = L, path.copy()
            elif w not in path:
                path.append(w); dfs(path); path.pop()
    dfs([start])
    return best['path']

def genetic_longest_cycle(G, start, pop_size=30, generations=100, mutation_rate=0.2):
    def random_path():
        visited = {start}
        path = [start]
        curr = start
        while True:
            cands = [w for w in G.neighbors(curr) if w not in visited]
            if not cands: break
            nxt = random.choice(cands)
            visited.add(nxt)
            path.append(nxt)
            curr = nxt
        return path
    def fitness(path):
        return len(path) + (1 if start in G.neighbors(path[-1]) else 0)
    # inicializa população
    pop = [random_path() for _ in range(pop_size)]
    best = max(pop, key=fitness)
    for _ in range(generations):
        # seleção por torneio
        new_pop = []
        for _ in range(pop_size):
            a, b = random.sample(pop, 2)
            new_pop.append(a if fitness(a) > fitness(b) else b)
        # crossover
        children = []
        for i in range(0, pop_size, 2):
            p1, p2 = new_pop[i], new_pop[i+1]
            common = list(set(p1[1:]) & set(p2[1:]))
            if common:
                c = random.choice(common)
                i1, i2 = p1.index(c), p2.index(c)
                child = p1[:i1] + p2[i2:]
                seen = set()
                child = [x for x in child if not (x in seen or seen.add(x))]
                children.append(child)
            else:
                children.extend([p1, p2])
        # mutação
        pop = []
        for path in children:
            if random.random() < mutation_rate:
                idx = random.randrange(1, len(path))
                visited = set(path[:idx+1])
                new_path = path[:idx+1]
                curr = new_path[-1]
                while True:
                    cands = [w for w in G.neighbors(curr) if w not in visited]
                    if not cands: break
                    nxt = random.choice(cands)
                    visited.add(nxt)
                    new_path.append(nxt)
                    curr = nxt
                path = new_path
            pop.append(path)
        # atualiza melhor
        cand = max(pop, key=fitness)
        if fitness(cand) > fitness(best):
            best = cand
    if start in G.neighbors(best[-1]):
        best.append(start)
    return best


#  ============================================
#  ======               WIP              ======
#  ============================================


def greedy_dfs_cycle(G, start):
    visited = {start}
    path = [start]
    while True:
        curr = path[-1]
        cands = [w for w in G.neighbors(curr) if w not in visited]
        if start in G.neighbors(curr) and len(path) > 1:
            path.append(start)
            return path
        if not cands:
            return path
        next_node = max(cands, key=lambda x: G.degree(x))
        visited.add(next_node)
        path.append(next_node)

def random_restart_greedy(G, start, iters=100):
    best = []
    for _ in range(iters):
        visited = {start}
        path = [start]
        while True:
            curr = path[-1]
            cands = [w for w in G.neighbors(curr) if w not in visited or (w == start and len(path) > 1)]
            if start in cands and len(path) > 1:
                path.append(start)
                break
            if not cands:
                break
            maxdeg = max(G.degree(w) for w in cands if w != start)
            top = [w for w in cands if G.degree(w) == maxdeg and w != start]
            next_node = random.choice(top)
            visited.add(next_node)
            path.append(next_node)
        if len(path) > len(best):
            best = path.copy()
    return best

def simulated_annealing_cycle(G, start, iters=1000, T0=1.0, alpha=0.995):
    curr = greedy_dfs_cycle(G, start)
    best = curr[:]
    curr_f = len(curr); best_f = curr_f
    T = T0
    for _ in range(iters):
        if len(curr) > 1:
            cut = random.randint(1, len(curr)-1)
            curr = curr[:cut]
        visited = set(curr)
        path = curr[:]
        while True:
            curr_node = path[-1]
            cands = [w for w in G.neighbors(curr_node) if w not in visited]
            if start in G.neighbors(curr_node) and len(path) > 1:
                path.append(start)
                break
            if not cands:
                break
            nxt = random.choice(cands)
            visited.add(nxt)
            path.append(nxt)
        f = len(path)
        if f > curr_f or random.random() < math.exp((f - curr_f) / T):
            curr, curr_f = path, f
            if f > best_f:
                best, best_f = path, f
        T *= alpha
    if best[-1] != start and start in G.neighbors(best[-1]):
        best.append(start)
    return best

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <dataset_file>")
        sys.exit(1)

    G = load_graph(sys.argv[1])
    start = input("Choose initial station (leave blank to test all): ").strip()

    if start and start not in G:
        print(f"No '{start}' station.")
        sys.exit(1)

    # Seleção de múltiplos algoritmos
    print("Select algorithm(s) to run (comma separated) ('all' to use all algorithms (NOT RECOMMENDED)): ")
    print("  1- Brute Force")
    print("  2- Branch-and-Bound")
    print("  3- Genetic Algorithm")
    print("  4- Greedy DFS (WIP - DO NOT USE)")
    print("  5- Greedy Random-Restart (WIP - DO NOT USE)")
    print("  6- Simulated Annealing (WIP - DO NOT USE)")
    choice = input("  > ").strip().lower()

    valid = {"1","2","3","4","5","6"}
    if choice in ("all"):
        algs = sorted(valid)
    else:
        algs = sorted(token for token in choice.replace(" ", "").split(",") if token in valid)

    if not algs:
        print("No valid option selected. Exiting.")
        sys.exit(1)

    funcs = {
        "1": brute_force_longest_cycle,
        "2": branch_and_bound_longest_cycle,
        "3": genetic_longest_cycle,
        "4": greedy_dfs_cycle,
        "5": random_restart_greedy,
        "6": simulated_annealing_cycle,
    }
    
    names = {
        "1": "Brute Force",
        "2": "Branch-and-Bound",
        "3": "Genetic Algorithm",
        "4": "Greedy DFS",
        "5": "Greedy Random-Restart",
        "6": "Simulated Annealing",
    }

    print("\n--- Longest Simple Path ---\n")
    for key in algs:
        func = funcs[key]
        name = names[key]

        startTime = time.perf_counter()
        if start:
            path = func(G, start)
            if path and path[-1] != start and start in G.neighbors(path[-1]):
                closed = path + [start]
            else:
                closed = path[:]
            length = len(closed)
            start_loc = start
        else:
            best = {'start': None, 'length': 0, 'path': []}
            for u in G.nodes():
                p = func(G, u)
                if p and p[-1] != u and u in G.neighbors(p[-1]):
                    p_closed = p + [u]
                else:
                    p_closed = p
                l = len(p_closed)
                if l > best['length']:
                    best = {'start': u, 'length': l, 'path': p}
            start_loc = best['start']
            if best['path'] and best['path'][-1] != start_loc and start_loc in G.neighbors(best['path'][-1]):
                closed = best['path'] + [start_loc]
            else:
                closed = best['path'][:]
            length = best['length']

        dur = time.perf_counter() - startTime

        print(f"{name}:")
        print(f"  Starting Location = {start_loc}")
        print(f"  Length            = {length}")
        print(f"  Path              = {closed}")
        print(f"  Time              = {dur:.3f} s\n")
