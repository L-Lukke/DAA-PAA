**Metro Graph Analysis Tool**

This Python application provides utilities to load and visualize a metro or transit network from plain-text station and line definitions, and to analyze its structure using graph algorithms such as minimum dominating sets and longest simple cycles.

---

## Features

1. **Graph Loading**

   * Parses station coordinates from a `stations.txt` file.
   * Reads line definitions (station-to-station connections) from a `lines.txt` file, including line IDs and colors.
   * Builds an undirected NetworkX graph with station nodes and edges annotated by line information.

2. **Visualization**

   * Uses Tkinter to render the network on a scalable canvas.
   * Draws colored line segments and labeled station nodes.

3. **Minimum 1‑Dominating Set**

   * **Brute‑force** search for the smallest set of stations that dominate the network.
   * **Branch‑and‑Bound** optimized search with lower‑bound pruning.
   * **Greedy** heuristic algorithm for fast approximations.

4. **Longest Simple Cycle**
   Multiple strategies to find or approximate the longest simple cycle (starting and ending at a chosen station):

   * Brute‑force DFS
   * Branch‑and‑Bound with upper‑bound pruning
   * Genetic algorithm
   * Greedy depth‑first search
   * GRASP (Greedy Random‑Restart)
   * Simulated Annealing (best for complete graphs) (WIP)
   * Tabu Search (WIP)
   * Ant Colony Optimization (WIP)

5. **Command‑Line Interface**

   * Menu-driven interface:

     1. Draw the graph
     2. Compute minimum dominating set(s)
     3. Compute longest simple cycle(s)
     4. Exit
   * Supports running individual or all algorithms, timing their execution.

---

## Installation and Requirements

* Python 3.7+
* Required packages:

  ```bash
  pip install networkx
  ```
* Tkinter is used for drawing; ensure your Python installation includes it (usually bundled).

---

## Usage

1. Prepare your data:

   * **stations.txt**: each line in the format:

     ```plaintext
     StationName  X  Y
     ```

     where `X` and `Y` are integer coordinates.
   * **lines.txt**: blocks separated by blank lines. Each block starts with:

     ```plaintext
     <LineID>  <Color>  <Count>
     ```

     followed by `Count` lines of `StationA  StationB` pairs.

2. Run the script:

   ```bash
   python main.py stations.txt lines.txt
   ```

3. Follow the on‑screen menu prompts to:

   * Draw the graph window
   * Select algorithms for dominating sets or longest cycle searches
   * View results and timings in the console

---

## Code Structure

* **Graph I/O**

  * `load_stations(path)`
  * `load_lines(path)`
  * `load_graph(stations_path, lines_path)`
  * `build_visual_graph(stations, lines)`

* **Visualization**

  * `draw_on_canvas(G, stations, size=(920,920), node_r=8)`

* **Dominating Set Algorithms**

  * `brute_force_min_dominating_set(G)`
  * `branch_and_bound_min_dominating_set(G)`
  * `greedy_dominating_set(G)`

* **Longest Cycle Algorithms**

  * `brute_force_longest_cycle(G, start)`
  * `branch_and_bound_longest_cycle(G, start)`
  * `genetic_longest_cycle(G, start, pop_size=20, generations=200, mutation_rate=0.2)`
  * `greedy_dfs_cycle(G, start)`
  * `grasp(G, start, k=3, iterations=10000)`
  * `simulated_annealing_longest_cycle(G, start, T0=1.0, Tmin=1e-4, alpha=0.9, max_iters=1000)`
  * `tabu_search_longest_cycle(G, start, tabu_tenure=10, max_iters=100, neigh_sample=50)`
  * `aco_longest_cycle(G, start, num_ants=20, num_iters=100, alpha=1.0, beta=2.0, rho=0.1, Q=1.0)`

* **Main Menu**

  * `main()` handles argument parsing and CLI interaction.

---

## Notes

* Some algorithms (brute‑force, branch‑and‑bound) can be very slow on large networks.
* The simulated annealing implementation assumes a complete graph; use with caution on sparse networks.
* Simulated Annealing, Tabu Search and Ant Colony are poorly commented and implemented, I really don't know if they actually work.