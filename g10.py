import sys
import time
import math
import itertools
import random
from collections import deque
import networkx as nx
import tkinter as tk
import networkx as nx

def load_graph(stations_path, lines_path):
    G = nx.Graph()
    
    with open(stations_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            name, xs, ys = line.split()
            x, y = int(xs), int(ys)
            G.add_node(name, pos=(x, y))
    
    with open(lines_path, 'r', encoding='utf-8') as f:
        while True:
            header = f.readline()
            if not header:
                break  # EOF
            header = header.strip()
            if not header:
                continue  # skip blank lines
            parts = header.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid line header: {header!r}")
            line_id, color, count_s = parts[:3]
            try:
                count = int(count_s)
            except ValueError:
                raise ValueError(f"Invalid connection count in header: {header!r}")
            # read exactly `count` connections
            for _ in range(count):
                conn = f.readline()
                if not conn:
                    raise EOFError("Unexpected end of file reading connections")
                u, v = conn.strip().split()
                if not G.has_node(u) or not G.has_node(v):
                    raise KeyError(f"Station {u!r} or {v!r} not found in stations file")
                # add edge; if multiple lines share the same pair, store as list
                if G.has_edge(u, v):
                    # append to existing attributes
                    G.edges[u, v].setdefault('lines', []).append(line_id)
                    G.edges[u, v].setdefault('colors', []).append(color)
                else:
                    G.add_edge(u, v, line=line_id, color=color, lines=[line_id], colors=[color])
    
    return G

# ----------- Map Drawing  -----------
def load_stations(path):
    stations = {}
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            name, xs, ys = line.split()
            stations[name] = (float(xs), float(ys))
    return stations

def load_lines(path):
    lines = []
    current = None
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                current = None
                continue
            parts = line.split()
            if parts[0].lower().startswith('line'):
                current = {'name': parts[0], 'color': parts[1], 'edges': []}
                lines.append(current)
            else:
                a, b = parts[:2]
                current['edges'].append((a, b))
    return lines

def build_visual_graph(stations, lines):
    G = nx.Graph()
    for name, pos in stations.items():
        G.add_node(name, pos=pos)
    for line in lines:
        for a, b in line['edges']:
            if a in stations and b in stations:
                G.add_edge(a, b, color=line['color'], line=line['name'])
            else:
                print(f"Warning: skipping edge {a}-{b} (unknown station)")
    return G

def draw_on_canvas(G, stations, size=(920,920), node_r=8):
    xs = [p[0] for p in stations.values()]
    ys = [p[1] for p in stations.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 50
    sx = (size[0]-2*pad)/(maxx-minx) if maxx>minx else 1
    sy = (size[1]-2*pad)/(maxy-miny) if maxy>miny else 1

    def cvt(p):
        x, y = p
        cx = pad + (x-minx)*sx
        cy = pad + (y - miny) * sy
        return cx, cy

    root = tk.Tk()
    root.title("Map")
    canvas = tk.Canvas(root, width=size[0], height=size[1], bg='white')
    canvas.pack()
    for u, v, data in G.edges(data=True):
        cu, cv = cvt(stations[u]), cvt(stations[v])
        canvas.create_line(cu[0], cu[1], cv[0], cv[1], fill=data['color'], width=4)
    for name, pos in stations.items():
        cx, cy = cvt(pos)
        canvas.create_oval(cx-node_r, cy-node_r, cx+node_r, cy+node_r, fill='white', outline='black')
        canvas.create_text(cx, cy-node_r-2, text=name, anchor=tk.S, font=('TkDefaultFont',8))
    root.mainloop()

# ----------- Minimum 1‐Dominating Set -----------
def brute_force_min_dominating_set(G): # note! this only returns the first minimum dominating set it finds, not all of them (if there are more than one)
    for r in range(1, len(G)+1): # r is the size of the node groups that possibly dominate G (worst case edgeless graph this is n nodes)
        for combo in itertools.combinations(G.nodes(), r): # this is the part where it explodes, as it iterate through every different combination of n packaged in size r groups
            domd = set()
            for u in combo:
                domd |= {u} | set(G.neighbors(u)) # take every neighbour of every node in 'combo' (including themselves) and shove them into a set
            if len(domd) == len(G): # if the lengyh of all nodes in 'combo' summed with all of its neighbours is equal to the size of the graph, it is a min. dom. set
                return set(combo) # this actually works because sets can't have duplicates

def branch_and_bound_min_dominating_set(G):
    nodes = list(G.nodes()); n = len(nodes)
    neighs = {u:{u}|set(G.neighbors(u)) for u in nodes}
    delta = max(len(neighs[u]) for u in nodes)
    best = {'size':n+1, 'set':set()}
    def dfs(dom, domd):
        if len(dom) >= best['size']:
            return
        if len(domd) == n:
            best['size'], best['set'] = len(dom), dom.copy()
            return
        lb = math.ceil((n - len(domd)) / delta)
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
    while len(domd) < len(G):
        u = max(G.nodes(), key=lambda x: len(({x}|set(G.neighbors(x))) - domd))
        dom.add(u)
        domd |= {u} | set(G.neighbors(u))
    return dom

# ----------- Longest Simple Cycle -----------

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

    if best_path and start in G.neighbors(best_path[-1]):
        best_path.append(start)
    
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
    
    dfs([start])    
     
    return best['path']

def genetic_longest_cycle(G, start, pop_size=30, generations=100, mutation_rate=0):
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
    
    return None

def grasp(G, start, k=3, iterations=10000):
    best_cycle = []
    
    for _ in range(iterations):
        visited = {start}
        path = [start]
        cycle = None
        
        while True:
            curr = path[-1]
            # Unvisited neighbors
            cands = [w for w in G.neighbors(curr) if w not in visited]
            
            # If no forward move, attempt to close cycle back to start
            if not cands:
                if start in G.neighbors(curr) and len(path) >= 3:
                    cycle = path + [start]
                break
            
            # Build RCL of top-k neighbors by degree
            cands.sort(key=lambda x: G.degree(x), reverse=True)
            rcl = cands[:k]
            
            # Randomly pick next node from RCL
            nxt = random.choice(rcl)
            visited.add(nxt)
            path.append(nxt)
        
        # Update best cycle if this iteration found a longer one
        if cycle and len(cycle) > len(best_cycle):
            best_cycle = cycle
    
    return best_cycle if best_cycle else None

def greedy_dfs_cycle(G, start):
    visited = {start}
    path = [start]

    while True:
        curr = path[-1]

        cands = [w for w in G.neighbors(curr) if w not in visited]
        if cands:
            nxt = max(cands, key=lambda w: G.degree(w))
            visited.add(nxt)
            path.append(nxt)
            continue

        for w in G.neighbors(curr):
            if w in visited:
                idx = path.index(w)
                if len(path) - idx + 1 >= 3:
                    return path[idx:] + [w]

        raise ValueError(f"No cycle found from start={start} via greedy walk")


def simulated_annealing_longest_cycle(G, start, T0=1.0, Tmin=1e-4, alpha=0.995, max_iters=10000):
    def random_closable_cycle():
        while True:
            path = [start]
            visited = {start}
            u = start
            while True:
                nbrs = [v for v in G.neighbors(u) if v not in visited]
                if not nbrs: break
                v = random.choice(nbrs)
                path.append(v); visited.add(v); u = v
            if start in G.neighbors(u) and len(path) > 2:
                return path + [start]

    def cycle_length(cycle):
        return len(cycle) - 1

    def two_opt(cycle):
        n = len(cycle) - 1
        i, k = sorted(random.sample(range(1, n), 2))
        new_cycle = cycle[:i] + cycle[i:k+1][::-1] + cycle[k+1:]
        return new_cycle

    def node_insert(cycle):
        if len(cycle) <= 4:
            return cycle[:]
        i, j = random.sample(range(1, len(cycle)-1), 2)
        c = cycle[:]
        node = c.pop(i)
        c.insert(j, node)
        return c

    def node_swap(cycle):
        if len(cycle) <= 4:
            return cycle[:]
        i, j = random.sample(range(1, len(cycle)-1), 2)
        c = cycle[:]
        c[i], c[j] = c[j], c[i]
        return c

    def random_neighbor(cycle):
        move = random.choice([two_opt, node_insert, node_swap])
        nbr = move(cycle)
        if len(set(nbr[:-1])) == len(nbr)-1 and nbr[0] == nbr[-1]:
            return nbr
        return cycle

    current = random_closable_cycle()
    best = current[:]
    T = T0

    for it in range(max_iters):
        if T < Tmin:
            break
        candidate = random_neighbor(current)
        delta = cycle_length(candidate) - cycle_length(current)
        if delta >= 0 or random.random() < math.exp(delta / T):
            current = candidate
            if cycle_length(current) > cycle_length(best):
                best = current[:]
        T *= alpha

    return best


def tabu_search_longest_cycle(G, start, tabu_tenure=10, max_iters=1000, neigh_sample=50):
    def random_closable_cycle():
        while True:
            path = [start]; vis={start}; u=start
            while True:
                nbrs=[v for v in G.neighbors(u) if v not in vis]
                if not nbrs: break
                v=random.choice(nbrs); path.append(v); vis.add(v); u=v
            if start in G.neighbors(u) and len(path)>2:
                return path+[start]

    def cycle_length(cycle):
        return len(cycle)-1

    def neighbors(cycle):
        nbrs=[]
        n=len(cycle)-1
        for _ in range(neigh_sample):
            i,k=sorted(random.sample(range(1,n),2))
            c2=cycle[:i]+cycle[i:k+1][::-1]+cycle[k+1:]
            if len(set(c2[:-1]))==len(c2)-1:
                nbrs.append(("2-opt", i, k, c2))
        for _ in range(neigh_sample//2):
            i,j=random.sample(range(1,n),2)
            c2=cycle[:]; c2[i],c2[j]=c2[j],c2[i]
            if len(set(c2[:-1]))==len(c2)-1:
                nbrs.append(("swap", i, j, c2))
        return nbrs

    current = random_closable_cycle()
    best_global = current[:]
    tabu = deque(maxlen=tabu_tenure)

    for it in range(max_iters):
        cand_moves = neighbors(current)
        filtered = []
        for m,i,j,c in cand_moves:
            move_id = (m, i, j)
            if move_id not in tabu or cycle_length(c) > cycle_length(best_global):
                filtered.append((move_id, c))
        if not filtered:
            break
        move_id, best_nb = max(filtered, key=lambda x: cycle_length(x[1]))
        tabu.append(move_id)
        current = best_nb
        if cycle_length(current) > cycle_length(best_global):
            best_global = current[:]

    return best_global

def aco_longest_cycle(G, start, num_ants=20, num_iters=100, alpha=1.0, beta=2.0, rho=0.1, Q=1.0):
    tau = {u: {v: 1.0 for v in G[u]} for u in G}

    def heuristic(u, v):
        return 1.0 / (G.degree(v) or 1)

    def construct_cycle():
        path = [start]; visited={start}; u = start
        while True:
            nbrs = [v for v in G.neighbors(u) if v not in visited]
            closers = [v for v in G.neighbors(u) if v == start and len(path) > 2]
            choices = nbrs + closers
            if not choices:
                break
            probs = []
            for v in choices:
                p = (tau[u][v]**alpha) * (heuristic(u, v)**beta)
                probs.append(p)
            s = sum(probs)
            r = random.random() * s
            cum = 0
            for v, p in zip(choices, probs):
                cum += p
                if r <= cum:
                    nxt = v
                    break
            if nxt == start:
                return path + [start]
            path.append(nxt); visited.add(nxt); u = nxt
        return None

    best_cycle = None

    for it in range(num_iters):
        all_cycles = []
        for _ in range(num_ants):
            c = construct_cycle()
            if c is not None:
                all_cycles.append(c)
        if not all_cycles:
            continue
        best_it = max(all_cycles, key=lambda c: len(c))
        if best_cycle is None or len(best_it) > len(best_cycle):
            best_cycle = best_it[:]
        for u in tau:
            for v in tau[u]:
                tau[u][v] *= (1 - rho)
        L = len(best_cycle) - 1
        for u, v in zip(best_cycle[:-1], best_cycle[1:]):
            tau[u][v] += Q / L
            tau[v][u] += Q / L

    # this sux very bad
    if best_cycle is None:
        for v in G.neighbors(start):
            return [start, v, start]
        return []

    return best_cycle

# ----------- Menu -----------

def main():
    if len(sys.argv) != 3:
        print(f"Usagw: python {sys.argv[0]} <stations.txt> <lines.txt>")
        sys.exit(1)

    sf = sys.argv[1]
    stations = load_stations(sf)
    G = load_graph(sys.argv[1], sys.argv[2])

    while True:
        print("Select functionality:")
        print("  1- Draw Graph")
        print("  2- Minimum 1-Dominating Set")
        print("  3- Cyclic Longest Simple Path")
        print("  4- Exit")
        choice = input(" > ").strip()
        
        if choice == '1':
            draw_on_canvas(G, stations)
            print("")

        elif choice == '2':
            print("\nSelect algorithm(s) ('all' to run all) (comma separated):")
            print("  1- Brute Force (this may take forever)")
            print("  2- Branch-and-Bound (this may take *almost* forever)")
            print("  3- Greedy")
            algs = input(" > ").strip().lower()

            if algs == 'all':
                sel = {'1','2','3'}
            else:
                sel = set(token for token in algs.replace(' ','').split(',') if token in ('1','2','3'))
            if not sel:
                print("No valid choice. Exiting.")
                sys.exit(1)

            print("\n--- Minimum 1-Dominating Set Results ---\n")

            if "1" in sel:
                start = time.perf_counter()
                bf_dom = brute_force_min_dominating_set(G)
                dur = time.perf_counter() - start
                print(f"Brute Force:")
                print(f"  Size = {len(bf_dom)}")
                print(f"  Set  = {bf_dom}")
                print(f"  Time = {dur:.3f} s\n")

            if "2" in sel:
                start = time.perf_counter()
                bnb_dom = branch_and_bound_min_dominating_set(G)
                dur = time.perf_counter() - start
                print(f"Branch-and-Bound:")
                print(f"  Size = {len(bnb_dom)}")
                print(f"  Set  = {bnb_dom}")
                print(f"  Time = {dur:.3f} s\n")

            if "3" in sel:
                start = time.perf_counter()
                greedy_dom = greedy_dominating_set(G)
                dur = time.perf_counter() - start
                print(f"Greedt:")
                print(f"  Size = {len(greedy_dom)}")
                print(f"  Set  = {greedy_dom}")
                print(f"  Time = {dur:.3f} s\n")
                

        elif choice == '3':
            start = input("\nStarting node (blank to auto-select best (this may take a while)): ").strip()

            if start and start not in G:
                print(f"Node '{start}' not in graph.")
                sys.exit(1)

            print("Select algorithm(s) to run (comma separated) ('all' run all (NOT RECOMMENDED)): ")
            print("  1- Brute Force")
            print("  2- Branch-and-Bound")
            print("  3- Genetic Algorithm")
            print("  4- Greedy DFS")
            print("  5- Greedy Random-Restart")
            print("  6- Simulated Annealing (WIP)")
            print("  7- Tabu Search (WIP)")
            print("  8- Ant Colony (WIP)")
            algs = input(" > ").strip().lower()

            valid = {"1","2","3","4","5","6","7","8"}

            if algs == "all":
                algs = sorted(valid)
            else:
                algs = sorted(token for token in algs.replace(" ", "").split(",") if token in valid)

            if not algs:
                print("No valid option selected. Exiting.")
                sys.exit(1)

            funcs = {
                "1": brute_force_longest_cycle,
                "2": branch_and_bound_longest_cycle,
                "3": genetic_longest_cycle,
                "4": greedy_dfs_cycle,
                "5": grasp,
                "6": simulated_annealing_longest_cycle,
                "7": tabu_search_longest_cycle,
                "8": aco_longest_cycle,
            }

            names = {
                "1": "Brute Force",
                "2": "Branch-and-Bound",
                "3": "Genetic Algorithm",
                "4": "Greedy DFS",
                "5": "Greedy Random-Restart",
                "6": "Simulated Annealing",
                "7": "Tabu Search",
                "8": "Ant Colony",
            }

            print("\n--- Longest Simple Path ---\n")

            for key in algs:
                func = funcs[key]
                name = names[key]

                startTime = time.perf_counter()
                if start:
                    path = func(G, start)
                    if path is None:
                        closed = []
                    elif path and path[-1] != start and start in G.neighbors(path[-1]):
                        closed = path + [start]
                    else:
                        closed = path[:]
                    length = len(closed)
                    start_loc = start
                else:
                    best = {'start': None, 'length': 0, 'path': []}
                    for u in G.nodes():
                        p = func(G, u)
                        if p is None:
                            continue
                        if p and p[-1] != u and u in G.neighbors(p[-1]):
                            p_closed = p + [u]
                        else:
                            p_closed = p
                        if p_closed:
                            l = len(p_closed)
                            if l > best['length']:
                                best = {'start': u, 'length': l, 'path': p}

                    start_loc = best['start']
                    bp = best['path'] or []
                    if bp and bp[-1] != start_loc and start_loc in G.neighbors(bp[-1]):
                        closed = bp + [start_loc]
                    else:
                        closed = bp[:]
                    length = len(closed)

                dur = time.perf_counter() - startTime

                print(f"{name}:")
                print(f"  Starting Location = {start_loc}")
                print(f"  Length            = {length}")
                print(f"  Path              = {closed}")
                print(f"  Time              = {dur:.3f} s\n")

        
        elif choice == '4':
            print("Exiting.\n")
            break

        else:
            print("Invalid option. Exiting.")
            sys.exit(1)

if __name__ == '__main__':
    main()
