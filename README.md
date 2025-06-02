# **Metro Graph Analysis Tool**

A Python application for visualizing and analyzing metro/transit networks. It includes both exact and heuristic graph algorithms to find minimum dominating sets and the longest simple cycles in the network.

## Features

### **1. Graph Loading**

* Parses station coordinates from a `stations.txt` file.
* Reads line definitions (connections) from a `lines.txt` file.
* Supports multi-line segments with color information.
* Builds a `NetworkX` undirected graph where:

  * Nodes represent stations (with position).
  * Edges include line IDs and color lists for visualization.

### **2. Visualization**

* Uses **Tkinter** to draw the metro map:

  * Colored edges by line.
  * Station names and coordinates are scaled to fit canvas.
* Interactive window displays the graph layout.

### **3. Minimum 1-Dominating Set**

Finds the smallest set of stations such that every station is either in the set or adjacent to one.

Algorithms available:

* `Brute Force`: Guaranteed minimum but slow for large graphs.
* `Branch and Bound`: Exact and faster using lower-bound pruning.
* `Greedy`: Fast heuristic approximation.

### **4. Longest Simple Cycle**

Finds or approximates the longest simple (non-repeating) cycle starting and ending at a chosen station.

Algorithms:

* `Brute Force DFS`
* `Branch and Bound`
* `Genetic Algorithm`
* `Greedy DFS`
* `GRASP (Greedy Random-Restart)`
* **(WIP / Experimental)**:

  * `Simulated Annealing`
  * `Tabu Search`
  * `Ant Colony Optimization`

Each method can be run individually or in combination via CLI.

### **5. Command-Line Interface**

Menu-driven CLI with these options:

```
1 - Draw Graph
2 - Minimum 1-Dominating Set
3 - Longest Simple Cycle
4 - Exit
```

Algorithm selection is interactive, supports running all or specific ones, and displays results with timing.

## Input File Format

### **stations.txt**

Each line:

```
StationName  X  Y
```

* X and Y are integer coordinates.

### **lines.txt**

Blocks separated by blank lines. Each block:

```
<LineID>  <Color>  <Count>
StationA  StationB
...
```

* `Count` lines follow, each representing a station-to-station connection.

## Installation

**Requirements:**

* Python 3.7+
* Packages:

  ```bash
  pip install networkx
  ```
* Tkinter (included with most Python installs)

## Usage

Run:

```bash
python g10.py stations.txt lines.txt
```

Follow the on-screen instructions to:

* View the graph
* Compute dominating sets
* Search for longest cycles

## Code Structure

### **Graph and I/O**

* `load_stations(path)`
* `load_graph(stations_path, lines_path)`

### **Visualization**

* `draw_on_canvas(G, stations, size=(920, 920), node_r=8)`

### **Dominating Set Algorithms**

* `brute_force_min_dominating_set(G)`
* `branch_and_bound_min_dominating_set(G)`
* `greedy_dominating_set(G)`

### **Longest Cycle Algorithms**

* `brute_force_longest_cycle(G, start)`
* `branch_and_bound_longest_cycle(G, start)`
* `genetic_longest_cycle(G, start)`
* `greedy_dfs_cycle(G, start)`
* `grasp(G, start)`
* *(WIP)* `simulated_annealing_longest_cycle(G, start)`
* *(WIP)* `tabu_search_longest_cycle(G, start)`
* *(WIP)* `aco_longest_cycle(G, start)`

### **Entry Point**

* `main()` handles command-line interface and routing.

## Notes & Warnings

* **Brute-force** methods are slow for graphs with more than \~10 nodes.
* **Simulated Annealing**, **Tabu Search**, and **Ant Colony** are labeled **WIP** — results may be unreliable.
* Graph must be connected and have valid formatting for meaningful results.
* Some pt-br color names are translated internally. (e.g., `'vermelho'` → `'red'`).
