import random

def genetic_longest_cycle(G, start, pop_size=50, generations=300, crossover_rate=0.8, mutation_rate=0.2):
    """    
    - Each initial individual is a random simple cycle that begins at `start`.

    Fitness function:
    - If the path is a closeable cycle, fitness = length of path.

    Evolutionary loop:
    1. Selection: pick two random individuals and keep the one with higher fitness.
    2. Crossover: pair up the selected individuals. For each pair:
       - Find common intermediate nodes (excluding start), pick one at random.
       - Create a children that inherits p1 up to that selected random node and p2 from that node onward.
       - Remove any duplicate vertices while preserving order. (oops?)
       - If no common node, inherit the fitter parent unchanged.
    3. Mutation: each child has a 20% chance of mutation:
       - Choose a random cut point in the path (not the start).
       - Truncate there, then perform a fresh random walk from that point to extend the path without revisiting vertices.
    4. Replacement: the new population becomes the set of mutated/unedited children.
    5. Track the best-ever individual across all generations.

    Special note from author: this kinda sucks bad for a single reason: mutation breaks established cycles, as it will destroy the last part of the path. 
    The fix would be to force mutation to complete the path, but this would suck even worse because it will cease to be a random walk and will now be a ton of
    "randm" walks until one of them closes the cycle, increasing the cost by a factor of mutation_rate*n² = O(n²) (i think). Idk how to fix it.
    
    Special note from author 2: this code permits inner cycles in the big cycle. Based on given instructions i really don't know if it should, but i dont think so.
    """

    def make_cycle_from_path(path):
        # if already closable, just append start
        if path and start in G.neighbors(path[-1]):
            return path + [start]
        # otherwise drop nodes until closable
        while len(path) > 1 and start not in G.neighbors(path[-1]):
            path.pop()
        if path and start in G.neighbors(path[-1]):
            return path + [start]
        return None

    def random_cycle():
        visited = {start}
        path = [start]
        curr = start
        # random-walk until stuck
        while True:
            cands = [w for w in G.neighbors(curr) if w not in visited]
            if not cands:
                break
            nxt = random.choice(cands)
            visited.add(nxt)
            path.append(nxt)
            curr = nxt
        # try to repair into a cycle
        cyc = make_cycle_from_path(path.copy())
        return cyc

    def fitness(cycle):
        if not cycle:
            return 0
        return len(cycle) - 1

    # initialize population: keep only valid cycles; if too few, fill with trivial 2-cycles (meh)
    pop = [random_cycle() for _ in range(pop_size)]
    neighs = list(G.neighbors(start))
    for i in range(len(pop)):
        if pop[i] is None:
            if neighs:
                v = random.choice(neighs)
                pop[i] = [start, v, start]
            else:
                pop[i] = None  # truly absolutely no cycle possible (special case in which start is completely disconnected from the graph)

    # track best ever
    best = max(pop, key=fitness)
    best_fit = fitness(best)

    for gen in range(generations):
        # Selection: Size-2 Tournament
        selected = []
        for _ in range(pop_size):
            a, b = random.sample(pop, 2)
            selected.append(a if fitness(a) >= fitness(b) else b) # FIGHT TO DEATH!

        # Crossover: Order Crossover excluding start from path
        children = []
        for i in range(0, pop_size, 2):
            p1, p2 = selected[i], selected[i+1]
            if random.random() < crossover_rate and p1 and p2:
                # chop `start` from the end of the path
                p1_mid = p1[1:-1]
                p2_mid = p2[1:-1]
                n = len(p1_mid)
                if n > 2:
                    a, b = sorted(random.sample(range(1, n), 2)) # start from 1, excludig 0 (always `start`)
                    seg = p1_mid[a:b]
                    tail = [v for v in p2_mid if v not in seg]
                    child_mid = tail[:a] + seg + tail[a:]
                else:
                    child_mid = p1_mid[:]  # too small, skip
                # repair & close
                child = make_cycle_from_path(child_mid)
                if child is None:
                    # fallback to fitter parent
                    child = p1 if fitness(p1) >= fitness(p2) else p2
            else:
                # no crossover: copy fitter parent
                child = p1 if fitness(p1) >= fitness(p2) else p2
            children.append(child)

        # Mutation: swap‐mutation on the cycle (excluding endpoints)
        new_pop = []
        for cyc in children:
            if cyc and random.random() < mutation_rate:
                core = cyc[1:-1]
                if len(core) >= 2:
                    i, j = sorted(random.sample(range(len(core)), 2))
                    core[i], core[j] = core[j], core[i]
                # repair & close
                mutated = make_cycle_from_path([start] + core)
                if mutated:
                    cyc = mutated
            new_pop.append(cyc)
        pop = new_pop

        # Update best
        for cyc in pop:
            f = fitness(cyc)
            if f > best_fit:
                best, best_fit = cyc, f

    return best if best_fit > 0 else None





import networkx as nx

def repair_path(seq, G, start):
    # seq: [S, …, X, …]
    repaired = [seq[0]]
    visited  = {seq[0]}
    for v in seq[1:]:
        u = repaired[-1]
        if G.has_edge(u, v):
            repaired.append(v)
            visited.add(v)
        else:
            # try to insert a shortest path from u → v
            try:
                sp = nx.shortest_path(G, u, v)
                # skip the first node (it's u), add the rest if they're new
                for w in sp[1:]:
                    if w in visited:
                        raise nx.NetworkXNoPath
                    repaired.append(w)
                    visited.add(w)
            except nx.NetworkXNoPath:
                # give up on linking to v; stop here
                break
    # now `repaired` is a simple path; trim to closable & close
    return make_cycle_from_path(repaired)
