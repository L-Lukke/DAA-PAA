import sys
import networkx as nx
from collections import deque
import math
import random
import time

def load_graph(path):
    G = nx.Graph()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("line"): # skip empty/header lines
                continue
            a, b = line.split()[:2] # expect at least two tokens per line: node_a node_b
            G.add_edge(a, b)
    return G


def brute_force_longest_cycle(G, start):
    """
    find the longest simple cycle starting and ending at any given node by exhaustive DFS:
    - maintains a global best_path list (constantly re-evaluated)
    - explores every possible path without revisiting nodes
    - checks for a return edge at each extension
    """

    best_path = []

    def dfs(path, visited):
        curr = path[-1]
        for w in G.neighbors(curr):
            if w == start and len(path) > 1: # found a cycle
                if len(path) > len(best_path): # its greater than the current best
                    best_path[:] = path[:]  # current path becomes best path
            elif w not in visited: # there is still a neighbour that wasn't visited - recurse: extend path to neighbour w
                visited.add(w)
                path.append(w)

                dfs(path, visited) # backtrack - removing the last step from the recursion (last node added to path and trying again with new neighbours (if available))

                path.pop()
                visited.remove(w)

    dfs([start], {start}) # initalization of dfs with given node
    return best_path


def branch_and_bound_longest_cycle(G, start):
    """
    find the longest simple cycle using branch-and-bound pruning:
    - precompute adjacency lists for speed (no need to G.neighbors(x) every recursion/loop)
    - reachable_count(u, vis) estimates how many additional nodes are reachable from u without revisiting those already in the current path
    - prune recursion when current_length + reachable_count <= best_length (in other words, when the current path will not result in beating the current best in its best case)
    """

    best = {'length': 0, 'path': []}
    adj = {u: list(G.neighbors(u)) for u in G.nodes()} # convert the relation node-neighbour to lists for every node

    def reachable_count(u, vis): # upper bound function: a simple BFS that counts all reachable nodes.

        seen = set(vis) # copy in all the nodes already on your current path, so you never count them again.
        q = deque([u]) # start a BFS (in the form of a double-ended queue, to pop the already explored nodes - in the left - and add its neighbours - in the right) from current node.
        seen.add(u)
        cnt = 0
        
        while q:
            x = q.popleft() # take the oldest node
            for y in adj[x]: # iterate through its neighbours
                if y not in seen:
                    seen.add(y)  # mark it so we don’t revisit it,
                    q.append(y)  # enqueue it for further exploration
                    cnt += 1 
        return cnt

    def dfs(path):
        curr = path[-1]

        if len(path) + reachable_count(curr, path) <= best['length']:
            return # if even the optimistic reachable count can't beat current best, stop recursion

        for w in adj[curr]:
            if w == start and len(path) > 1: # found a cycle
                if len(path) > best['length']: # its greater than the current best
                    best['length'] = len(path) # current length becomes best lenght
                    best['path'] = path.copy() # current path becomes best path
            elif w not in path:
                path.append(w)
                dfs(path) # backtrack - removing the last step from the recursion (last node added to path and trying again with new neighbours (if available))
                path.pop()

    dfs([start])
    return best['path']


def genetic_longest_cycle(G, start, pop_size=30, generations=100, mutation_rate=0.2):
    """
    heuristic search using a genetic algorithm:
    - randomly initialize a population of simple paths starting at a given `start` node.
    - fitness = path_length it can close to a cycle, else 0.
    - tournament selection of size 2.
    - single-point crossover at a common node.
    - mutation: truncate and randomly extend the path.
    - keep track of best solution over generations.
    """

    def random_path(): # generate a random simple path starting at `start` node by random walks.
        visited = {start}
        path = [start]
        curr = start
        while True:
            cands = [w for w in G.neighbors(curr) if w not in visited]
            if not cands:
                break
            nxt = random.choice(cands)  # choose next node of walk randomly among unvisited neighbour candidates
            visited.add(nxt)
            path.append(nxt)
            curr = nxt
        return path

    def fitness(path): # fitness equals path size for closable cycles.
        if start in G.neighbors(path[-1]):
            return len(path)
        else:
            return 0

    pop = [random_path() for _ in range(pop_size)] # initialize population of random walks of size `pop_size` (adjustable parameter)
    best = max(pop, key=fitness) # take best path (longest and/or closable)

    for _ in range(generations): # each iteration is a generation
        new_pop = []
        # new_pop array population will be done through size 2 tourament
        for _ in range(pop_size):
            a, b = random.sample(pop, 2) # takes 2 individuals a and b from current population
            new_pop.append(a if fitness(a) > fitness(b) else b) # compare them and select the better fit to append to new population

        children = []
        # creates children through crossover of 2 parents
        for i in range(0, pop_size, 2): # kind of assumes even array size? idk
            p1, p2 = new_pop[i], new_pop[i+1] # takes two parents, two consecutive elements in new_pop array
            common = list(set(p1[1:]) & set(p2[1:])) # find common intermediate nodes ((note that the starting point can not be included as it should be always equal)
            if common:
                c = random.choice(common) # take a random common corssover point
                i1, i2 = p1.index(c), p2.index(c)
                child = p1[:i1] + p2[i2:] # splice parents at the crossover point c
                seen = set() # create a set of seen nodes
                child = [x for x in child if not (x in seen or seen.add(x))] # use `seen` set to remove duplicates, each child node is only added if it is not in set 
                children.append(child)
            else:
                children.extend([p1, p2]) # this is the bad case, where there`s no crossover between parents, and so they are copied

        pop = [] # zeroes population array, as children is doing its job now (funny)
        # mutation and formation of new population (funny again)
        for path in children:
            if random.random() < mutation_rate: # 20% chance of mutation (parametrized, i just tried it a bunch of times and 20% was the sweetspot)
                idx = random.randrange(1, len(path)) # choose random cut point in the path
                visited = set(path[:idx+1]) # maintain visided list to aviod visiting the same node twice
                new_path = path[:idx+1] # create a new path that is equal to the previous one up until the cut point
                # now we just complete the path with a random walk
                curr = new_path[-1]
                while True:
                    cands = [w for w in G.neighbors(curr) if w not in visited]
                    if not cands:
                        break
                    nxt = random.choice(cands)
                    visited.add(nxt)
                    new_path.append(nxt)
                    curr = nxt
                path = new_path
            pop.append(path) # fills the population array with children (about 20% of them are mutated now)

        cand = max(pop, key=fitness) # update best individual
        if fitness(cand) > fitness(best):
            best = cand

    if start in G.neighbors(best[-1]): # close the cycle if possible
        best.append(start)
    return best

#  ===========================================
#  ======     not working fix later     ======
#  ===========================================

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

#  ===========================================
#  ======     not working fix later     ======
#  ===========================================

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

#  ===========================================
#  ======     not working fix later     ======
#  ===========================================

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


#  ===========================================
#  ======             main              ======
#  ===========================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <dataset_file>")
        sys.exit(1)

    G = load_graph(sys.argv[1])
    start = input("Choose initial station (leave blank to test all): ").strip()

    if start and start not in G:
        print(f"No '{start}' station in file.")
        sys.exit(1)

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