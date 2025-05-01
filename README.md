# Design and Analysis of Algorithms

## English

### Overview
This repository contains Python implementations for two graph problems on a metro network:

1. **Longest Simple Cycle**  
   Given a starting station, find the maximum‐length simple cycle (tourist route) that starts and ends at the same station without revisiting any intermediate station.

2. **Minimum 1-Dominating Set**  
   Find the minimum set of stations (kiosks) such that every station is at distance at most one from at least one kiosk.

These scripts were developed as part of the Design and Analysis of Algorithms course at PUC Minas.

---

### Requirements

- Python 3.7 or later  
- `networkx` library  

Install with:

```bash
pip install networkx
```

---

### Usage

#### 1. Longest Simple Path / Cycle

```bash
python longestSimplePath.py <dataset_file>
```

- **`<dataset_file>`**: Text file listing edges, one per line, in the format:
  ```
  StationA StationB
  ```
- The script will prompt for:
  1. **Starting station** (leave blank to test all stations).
  2. **Algorithm selection** (comma-separated list):
     1. Brute Force  
     2. Branch-and-Bound  
     3. Genetic Algorithm  
     4. Greedy DFS  
     5. Greedy Random-Restart
     6. Simulated Annealing 

For each selected algorithm it prints:

- Starting location  
- Cycle length  
- Path  
- Execution time  

---

#### 2. Minimum 1-Dominating Set

```bash
python minOneDominatingSet.py <dataset_file>
```

- **`<dataset_file>`**: Same format as above.
- The script will prompt for:
  1. **Algorithm selection**:
     1. Brute Force  
     2. Branch-and-Bound  
     3. Greedy Heuristic  

For each selected algorithm it prints:

- Dominating set size  
- Station set  
- Execution time  

---

### Input Format

- Stations are unique strings without spaces.  
- The dataset must list one edge per line:
  ```
  A B
  ```

---

### Algorithms Implemented

#### 1. Longest Simple Cycle
- **Brute Force**: DFS explores all simple paths, checks for cycle closure.  
- **Branch-and-Bound**: DFS with pruning via reachable‐node counts.  
- **Genetic Algorithm**: Random population, tournament selection, crossover, mutation.  
- **Greedy DFS**: At each step pick neighbor of highest degree.  
- **Greedy Random-Restart**: Repeat greedy with random restarts.  
- **Simulated Annealing**: Perturb paths, accept moves by Metropolis criterion.  

#### 2. Minimum 1-Dominating Set
- **Brute Force**: Try all combinations in increasing size until full coverage.  
- **Branch-and-Bound**: DFS with a lower bound based on maximum degree.  
- **Greedy Heuristic**: Iteratively pick the node covering the most uncovered neighbors.  

---

## Português

### Visão Geral
Este repositório contém implementações em Python para dois problemas de grafo na rede de metrô:

1. **Maior Ciclo Simples**  
   Dada uma estação inicial, encontra o ciclo simples de maior comprimento que inicia e termina na mesma estação sem revisitar estações intermediárias.

2. **Conjunto Dominante Mínimo 1**  
   Encontra o menor conjunto de estações (guichês) de modo que toda estação fique a no máximo uma aresta de distância de um guichê.

Desenvolvido para a disciplina “Projeto e Análise de Algoritmos” da PUC Minas.

---

### Requisitos

- Python 3.7 ou superior  
- Biblioteca `networkx`  

Instalação:

```bash
pip install networkx
```

---

### Uso

#### 1. Maior Ciclo Simples

```bash
python longestSimplePath.py <arquivo_de_dados>
```

- **`<arquivo_de_dados>`**: arquivo de texto com arestas no formato:
  ```
  EstacaoA EstacaoB
  ```
- O script solicitará:
  1. **Estação inicial** (deixe em branco para testar todas).  
  2. **Seleção de algoritmos** (lista separada por vírgulas):
     1. Força Bruta  
     2. Branch-and-Bound  
     3. Algoritmo Genético  
     4. Greedy DFS (em desenvolvimento – desaconselhado)  
     5. Greedy Random-Restart (em desenvolvimento – desaconselhado)  
     6. Simulated Annealing (em desenvolvimento – desaconselhado)  

Para cada algoritmo escolhido, exibe:

- Estação inicial  
- Comprimento do ciclo  
- Caminho  
- Tempo de execução  

---

#### 2. Conjunto Dominante Mínimo 1

```bash
python minOneDominatingSet.py <arquivo_de_dados>
```

- **`<arquivo_de_dados>`**: mesmo formato acima.  
- O script solicitará:
  1. **Seleção de algoritmos**:
     1. Força Bruta  
     2. Branch-and-Bound  
     3. Heurística Gulosa  

Para cada algoritmo escolhido, exibe:

- Tamanho do conjunto dominante  
- Conjunto de estações  
- Tempo de execução  

---

### Formato de Entrada

- Estaçõe são identificadas por strings únicas sem espaços.  
- Cada linha do arquivo deve conter uma aresta:
  ```
  A B
  ```

---

### Algoritmos Implementados

#### 1. Maior Ciclo Simples
- **Força Bruta**: DFS que explora todos os caminhos simples e verifica fechamento.  
- **Branch-and-Bound**: DFS com poda baseada em contagem de nós alcançáveis.  
- **Algoritmo Genético**: População inicial, torneio, crossover e mutação.  
- **Greedy DFS**: Escolhe vizinho de maior grau.  
- **Greedy Random-Restart**: Múltiplas reinicializações aleatórias.  
- **Simulated Annealing**: Perturba caminhos e usa critério de Metropolis.  

#### 2. Conjunto Dominante Mínimo 1
- **Força Bruta**: Testa todas as combinações até cobrir todo o grafo.  
- **Branch-and-Bound**: DFS com limite inferior baseado no grau máximo.  
- **Heurística Gulosa**: Escolhe iterativamente o nó que cobre mais vizinhos descobertos.  