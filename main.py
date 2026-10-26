import networkx as nx
import igraph as ig
import random
import numpy as np
import pandas as pd
import time
import os
import pickle
from bisect import bisect_right
from typing import List, Optional

class DCSA_BIM:
    def __init__(self, G: nx.Graph, n: int, B: float, alpha: float, beta: float, g_max: int, fl: float, AP: float,
                 costs: dict, efficiencies: dict, constraint_dict: dict, p: float = 0.01, core_dict: Optional[dict] = None):
        """
        Inicializa los parámetros del DCSA recibiendo las métricas ya precalculadas
        """
        self.G = G
        self.nodes = list(G.nodes())
        self.n = n
        self.B = B
        self.alpha = alpha
        self.beta = beta
        self.g_max = g_max
        self.fl = fl
        self.AP = AP
        self.p = p

        # Recibir las métricas precalculadas desde afuera
        self.costs = costs
        self.efficiencies = efficiencies
        self._constraint = constraint_dict
        self._core = core_dict

        # Listas de adyacencia y grados precalculados (más rápido que consultar networkx)
        self.adj = {v: list(G.neighbors(v)) for v in self.nodes}

        # Red en formato CSR (arreglos numpy) para calcular LIE de forma vectorizada
        self.idx = {v: i for i, v in enumerate(self.nodes)}
        self.deg_arr = np.array([len(self.adj[v]) for v in self.nodes], dtype=np.int64)
        self.indptr = np.zeros(len(self.nodes) + 1, dtype=np.int64)
        self.indptr[1:] = np.cumsum(self.deg_arr)
        self.indices = np.fromiter((self.idx[w] for v in self.nodes for w in self.adj[v]),
                                   dtype=np.int64, count=int(self.indptr[-1]))

        # Nodos ordenados por costo, para elegir rápidamente nodos que caben en un saldo
        self.nodes_by_cost = sorted(self.nodes, key=lambda v: self.costs[v])
        self.sorted_costs = [self.costs[v] for v in self.nodes_by_cost]
        self.min_cost = self.sorted_costs[0]

        self.candidates = [] # Se generan al inicio de optimize()
        self.lie_cache = {} # Memoria caché para no recalcular LIE

    def _get_cost(self, S: List[int]) -> float:
        """Calcula el costo total de un conjunto semilla"""
        return sum(self.costs[n] for n in S)

    def _random_node_within(self, limit: float, exclude) -> Optional[int]:
        """Retorna un nodo aleatorio con costo <= limit que no esté en exclude (None si no existe)"""
        idx = bisect_right(self.sorted_costs, limit)
        if idx == 0:
            return None
        # Muestreo con reintentos (rápido); si falla, se filtra explícitamente
        for _ in range(30):
            node = self.nodes_by_cost[random.randrange(idx)]
            if node not in exclude:
                return node
        valid = [v for v in self.nodes_by_cost[:idx] if v not in exclude]
        return random.choice(valid) if valid else None

    def _neighbors_of(self, rows: np.ndarray) -> np.ndarray:
        """Retorna los vecinos (con repeticiones) de los nodos con índices rows, sin bucles de Python"""
        lens = self.deg_arr[rows]
        ends = np.cumsum(lens)
        total = int(ends[-1]) if ends.size else 0
        offsets = np.repeat(self.indptr[rows] - (ends - lens), lens)
        return self.indices[offsets + np.arange(total)]

    def calculate_LIE(self, S: List[int]) -> float:
        """Estimación de Influencia Local (LIE) de 2 saltos con Caché"""
        if not S:
            return 0.0

        # Convertir la lista a un frozenset (inmutable) para usarlo como llave
        S_set = frozenset(S)
        if S_set in self.lie_cache:
            return self.lie_cache[S_set] # Si ya se calculó antes, retornar el valor al instante

        N = len(self.nodes)
        S_idx = np.fromiter((self.idx[v] for v in S_set), dtype=np.int64, count=len(S_set))

        # N1_S con la cantidad de vecinos en S de cada nodo
        neighbors_in_S = np.bincount(self._neighbors_of(S_idx), minlength=N)
        neighbors_in_S[S_idx] = 0
        N1_S = np.flatnonzero(neighbors_in_S)

        lie_value = float(len(S_set))

        if N1_S.size:
            inf_1 = float(np.sum(1 - (1 - self.p) ** neighbors_in_S[N1_S]))

            in_N2 = np.zeros(N, dtype=bool)
            in_N2[self._neighbors_of(N1_S)] = True
            in_N2[S_idx] = False
            in_N2[N1_S] = False
            N2_S = np.flatnonzero(in_N2)

            # d*_u: número de aristas del nodo u dentro de N1_S y N2_S
            sum_d = 0
            if N2_S.size:
                in_N12 = in_N2
                in_N12[N1_S] = True
                sum_d = int(np.count_nonzero(in_N12[self._neighbors_of(N2_S)]))

            lie_value += inf_1 + (inf_1 / N1_S.size) * self.p * sum_d

        # Guardar el resultado en el caché antes de retornar
        self.lie_cache[S_set] = lie_value
        return lie_value

    def generate_candidates(self, alpha_metric: float = 0.5, beta_cand: float = 3.0) -> List[int]:
        """Genera candidatos usando K-shell y Structural Holes"""
        k_shell = self._core if self._core is not None else nx.core_number(self.G)
        sum_ks = sum(k_shell.values())
        struct_holes = self._constraint
        sum_sh_inv = sum(1.0 / v if v > 0 else 0 for v in struct_holes.values())

        NC = {}
        for node in self.nodes:
            ks_term = k_shell[node] / sum_ks if sum_ks > 0 else 0
            sh_val = struct_holes[node]
            sh_term = (1.0 / sh_val) / sum_sh_inv if sh_val > 0 and sum_sh_inv > 0 else 0
            NC[node] = alpha_metric * ks_term + (1 - alpha_metric) * sh_term

        sorted_nodes = sorted(NC.keys(), key=lambda x: NC[x], reverse=True)

        avg_cost = sum(self.costs.values()) / len(self.nodes)
        k_est = max(1, int(self.B / avg_cost))
        top_beta_k = int(beta_cand * k_est)

        return sorted_nodes[:top_beta_k]

    def repair_solution(self, S: List[int]) -> List[int]:
        """
        Reparación en dos fases:
        1) Si excede B, elimina secuencialmente los nodos de menor eficiencia e(v) hasta cumplir la restricción
        2) Si el saldo restante lo permite, rellena con candidatos aleatorios cuyo costo no exceda el saldo
        """
        S_set = set(S)
        current_cost = self._get_cost(S_set)

        # Fase 1: eliminar nodos de peor eficiencia
        while current_cost > self.B:
            worst_node = min(S_set, key=lambda node: self.efficiencies[node])
            S_set.remove(worst_node)
            current_cost -= self.costs[worst_node]

        # Fase 2: rellenar con candidatos mientras quepan en el saldo
        remaining_budget = self.B - current_cost
        while remaining_budget >= self.min_cost:
            valid_candidates = [c for c in self.candidates if c not in S_set and self.costs[c] <= remaining_budget]
            if not valid_candidates:
                break
            new_node = random.choice(valid_candidates)
            S_set.add(new_node)
            remaining_budget -= self.costs[new_node]

        return list(S_set)

    def initialization(self) -> List[List[int]]:
        """Inicialización priorizando nodos en orden descendente según su eficiencia e(v)"""
        sorted_by_eff = sorted(self.nodes, key=lambda x: self.efficiencies[x], reverse=True)

        base_S = []
        base_cost = 0.0
        for node in sorted_by_eff:
            if base_cost + self.costs[node] <= self.B:
                base_S.append(node)
                base_cost += self.costs[node]

        X = []
        for i in range(self.n):
            xi = list(base_S)
            xi_set = set(xi)
            current_cost = base_cost
            for j in range(len(xi)):
                if random.random() > 0.5:
                    node_to_remove = xi[j]
                    cost_without = current_cost - self.costs[node_to_remove]
                    new_node = self._random_node_within(self.B - cost_without, xi_set)
                    if new_node is not None:
                        xi[j] = new_node
                        xi_set.remove(node_to_remove)
                        xi_set.add(new_node)
                        current_cost = cost_without + self.costs[new_node]
            X.append(xi)
        return X

    def update_position(self, x_i: List[int], m_j: List[int], r_i: float) -> List[int]:
        """Actualiza la posición y repara la solución"""
        x_new = list(x_i)
        x_new_set = set(x_new)
        m_j_set = set(m_j)

        for j in range(len(x_new)):
            intersection_val = 0 if x_new[j] in m_j_set else 1
            factor = r_i * self.fl * intersection_val
            H = 1 if factor >= 1 else 0

            if H == 1:
                while True:
                    new_node = random.choice(self.nodes)
                    if new_node not in x_new_set:
                        x_new_set.remove(x_new[j])
                        x_new_set.add(new_node)
                        x_new[j] = new_node
                        break

        return self.repair_solution(x_new)

    def random_walk(self, Candidates: List[int]) -> List[int]:
        """Caminata aleatoria utilizando candidatos hasta llenar el presupuesto B"""
        x_new = []
        current_cost = 0.0
        shuffled_cands = random.sample(Candidates, len(Candidates))

        for temp in shuffled_cands:
            if current_cost + self.costs[temp] <= self.B:
                x_new.append(temp)
                current_cost += self.costs[temp]

        return x_new

    def local_search(self, x_star: List[int]) -> List[int]:
        """
        Búsqueda local truncada: evalúa la sustitución de los nodos de menor eficiencia
        por los mejores vecinos posibles, respetando el presupuesto B
        """
        x_best = list(x_star)
        x_best_set = set(x_best)
        current_lie = self.calculate_LIE(x_best)
        current_cost = self._get_cost(x_best)

        # Ordenar por eficiencia ascendente
        sorted_x = sorted(x_best, key=lambda n: self.efficiencies[n])

        for node in sorted_x:
            neighbors = self.adj[node]

            # Búsqueda Local Truncada.
            # Si el nodo tiene demasiados vecinos, evalúa solo a los 15 más eficientes
            if len(neighbors) > 15:
                neighbors = sorted(neighbors, key=lambda n: self.efficiencies[n], reverse=True)[:15]

            for neighbor in neighbors:
                if neighbor not in x_best_set:
                    cost_diff = self.costs[neighbor] - self.costs[node]

                    if current_cost + cost_diff <= self.B:
                        x_temp = list(x_best)
                        idx = x_temp.index(node)
                        x_temp[idx] = neighbor

                        new_lie = self.calculate_LIE(x_temp)
                        if new_lie > current_lie:
                            x_best = x_temp
                            x_best_set.remove(node)
                            x_best_set.add(neighbor)
                            current_lie = new_lie
                            current_cost += cost_diff
                            node = neighbor # Se actualiza para que las siguientes comparaciones sean contra el nuevo
        return x_best

    def optimize(self):
        """Ejecuta el ciclo principal del DCSA."""
        Candidates = self.generate_candidates()
        self.candidates = Candidates

        X = self.initialization()
        M = [list(xi) for xi in X]

        # Fitness guardado junto a cada cuervo para no recalcularlo
        fit_X = [self.calculate_LIE(xi) for xi in X]
        fit_M = [self.calculate_LIE(mi) for mi in M]

        best_idx = int(np.argmax(fit_M))
        x_star = list(M[best_idx])
        fit_star = fit_M[best_idx]

        for iter_num in range(self.g_max):
            for i in range(self.n):
                j = random.randint(0, self.n - 1)
                r_i = random.random()
                r_j = random.random()

                if r_j >= self.AP:
                    # Seguimiento: la nueva posición se acepta siempre (la memoria guarda la mejor)
                    X[i] = self.update_position(X[i], M[j], r_i)
                    fit_X[i] = self.calculate_LIE(X[i])
                else:
                    # Caminata aleatoria: se acepta solo si mejora
                    x_new = self.random_walk(Candidates)
                    fit_new = self.calculate_LIE(x_new)
                    if fit_new > fit_X[i]:
                        X[i] = x_new
                        fit_X[i] = fit_new

            # Actualizar la memoria de los cuervos
            for i in range(self.n):
                if fit_X[i] > fit_M[i]:
                    M[i] = list(X[i])
                    fit_M[i] = fit_X[i]

            best_idx = int(np.argmax(fit_M))
            if fit_M[best_idx] > fit_star:
                x_star = list(M[best_idx])
                fit_star = fit_M[best_idx]

            x_star_prime = self.local_search(x_star)
            fit_prime = self.calculate_LIE(x_star_prime)
            if fit_prime > fit_star:
                x_star = x_star_prime
                fit_star = fit_prime

            print(f"Iteración {iter_num + 1}/{self.g_max} | Mejor LIE: {fit_star:.4f} | Nodos: {len(x_star)} | Costo: {self._get_cost(x_star):.2f}/{self.B:.2f}")

        return x_star

def calcular_spread_monte_carlo(G, semillas, p=0.01, iteraciones=10000):
    """Ejecuta el modelo IC múltiples veces para calcular el spread esperado (promedio)"""
    # Listas de adyacencia precalculadas (más rápido que G.neighbors, misma secuencia aleatoria)
    adj = {v: list(G.neighbors(v)) for v in G.nodes()}
    rand = random.random

    spread_total = 0
    for _ in range(iteraciones):
        nodos_activos = set(semillas)
        nuevos_activos = set(semillas)

        while nuevos_activos:
            proximos_activos = set()
            for u in nuevos_activos:
                for v in adj[u]:
                    if v not in nodos_activos and v not in proximos_activos:
                        if rand() < p:
                            proximos_activos.add(v)
            nuevos_activos = proximos_activos
            nodos_activos.update(proximos_activos)

        spread_total += len(nodos_activos)

    return spread_total / iteraciones

def cargar_metricas(G, ruta_red, carpeta_cache="Cache"):
    """
    Calcula Constraint, Betweenness y K-shell de la red, o los carga desde disco
    si ya fueron calculados antes para esta misma red
    """
    nombre = os.path.splitext(os.path.basename(ruta_red))[0]
    ruta_cache = os.path.join(carpeta_cache, f"{nombre}_metricas.pkl")

    if os.path.exists(ruta_cache):
        print(f"- Cargando métricas desde {ruta_cache}...")
        with open(ruta_cache, "rb") as f:
            metricas = pickle.load(f)
        if metricas["n_nodos"] == G.number_of_nodes() and metricas["n_aristas"] == G.number_of_edges():
            return metricas["constraint"], metricas["bc"], metricas["core"]
        print("- La caché no coincide con la red cargada, se recalcula...")

    # 1. igraph y Constraint (Structural Holes)
    print("- Construyendo igraph y calculando Constraint...")
    H = ig.Graph.from_networkx(G)
    constraints_list = H.constraint()
    constraint_dict = dict(zip(H.vs["_nx_name"], constraints_list))

    # 2. Betweenness Centrality
    print("- Calculando Betweenness Centrality...")
    bc_values = H.betweenness(directed=False, normalized=True, vertices=None, cutoff=-1)
    bc = dict(zip(H.vs["_nx_name"], bc_values))

    # 3. K-shell
    print("- Calculando K-shell...")
    core = nx.core_number(G)

    os.makedirs(carpeta_cache, exist_ok=True)
    with open(ruta_cache, "wb") as f:
        pickle.dump({"n_nodos": G.number_of_nodes(), "n_aristas": G.number_of_edges(),
                     "constraint": constraint_dict, "bc": bc, "core": core}, f)

    return constraint_dict, bc, core

if __name__ == "__main__":

    N_RUNS = 10

    # Cargar la red
    formato = input("Formato archivo (txt/csv): ")
    if formato == "txt":
        print("Cargando la red desde txt...")
        ruta_red = "Redes/facebook_combined.txt"
        G = nx.read_edgelist(ruta_red, comments='#', nodetype=int, create_using=nx.Graph())
    else:
        print("Cargando la red desde csv...")
        ruta_red = "Redes/musae_PTBR_edges.csv"
        df = pd.read_csv(ruta_red)
        G = nx.from_pandas_edgelist(df, source='nodo_1', target='nodo_2', create_using=nx.Graph())

    print(f"Red cargada: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

    # =====================================================================
    # BLOQUE DE PRE-CÁLCULOS (SE EJECUTA UNA SOLA VEZ, SE GUARDA EN DISCO)
    # =====================================================================
    print("\nIniciando pre-cálculo de métricas estructurales de la red...")
    start_precalc = time.time()

    constraint_dict, bc, core_dict = cargar_metricas(G, ruta_red)
    bc_min, bc_max = min(bc.values()), max(bc.values())

    # Grados, Costos y Eficiencias
    print("- Calculando Grados, Costos y Eficiencias...")
    degrees = dict(G.degree())
    deg_min, deg_max = min(degrees.values()), max(degrees.values())

    alpha_costo = 0.5
    beta_eficiencia = 0.5

    costs_dict = {}
    efficiencies_dict = {}

    for node in G.nodes():
        deg_norm = (degrees[node] - deg_min) / (deg_max - deg_min) if deg_max != deg_min else 0.0
        bc_norm = (bc[node] - bc_min) / (bc_max - bc_min) if bc_max != bc_min else 0.0

        c_v = 1.0 + alpha_costo * deg_norm
        e_v = (beta_eficiencia * deg_norm + (1.0 - beta_eficiencia) * bc_norm) / c_v

        costs_dict[node] = c_v
        efficiencies_dict[node] = e_v

    tiempo_precalc = time.time() - start_precalc
    print(f"Pre-cálculos finalizados en {tiempo_precalc:.2f} segundos.")
    # =====================================================================

    # Acumuladores
    resultados_bim = {"B": [], "tiempo": [], "LIE": [], "spread": [], "tam_semilla": []}

    for run in range(1, N_RUNS + 1):
        print(f"\n{'='*50}")
        print(f"RUN {run}/{N_RUNS}")
        print(f"{'='*50}")

        # Semilla fija por run para que los resultados sean reproducibles
        random.seed(run)
        np.random.seed(run)

        # --- DCSA-BIM ---
        # Pasar diccionarios precalculados a la clase
        dcsa_bim = DCSA_BIM(
            G=G, n=30, B=50, alpha=alpha_costo, beta=beta_eficiencia, g_max=100, fl=2.0, AP=0.15,
            costs=costs_dict,
            efficiencies=efficiencies_dict,
            constraint_dict=constraint_dict,
            core_dict=core_dict
        )

        start = time.time()
        semillas_bim = dcsa_bim.optimize()
        tiempo_bim = time.time() - start

        # Para el cálculo de B se usa el diccionario precargado
        B_bim = sum(costs_dict[n] for n in semillas_bim)
        LIE_bim = dcsa_bim.calculate_LIE(semillas_bim)

        # Cálculo de Monte Carlo
        spread_bim = calcular_spread_monte_carlo(G, semillas_bim, p=0.01, iteraciones=10000)

        resultados_bim["B"].append(B_bim)
        resultados_bim["tiempo"].append(tiempo_bim)
        resultados_bim["LIE"].append(LIE_bim)
        resultados_bim["spread"].append(spread_bim)
        resultados_bim["tam_semilla"].append(len(semillas_bim))

        print(f"[DCSA-BIM]  B={B_bim:.2f} | Tiempo={tiempo_bim:.2f}s | LIE={LIE_bim:.4f} | Spread={spread_bim:.2f}")

    # --- RESUMEN FINAL ---
    print(f"\n{'='*70}")
    print(f"RESUMEN PROMEDIO ({N_RUNS} runs)")
    print(f"{'='*70}")

    metricas = ["B", "tiempo", "LIE", "spread", "tam_semilla"]

    print(f"\n{'Métrica':<12} {'DCSA-BIM':>18}")
    print("-" * 50)
    for m in metricas:
        avg_bim   = np.mean(resultados_bim[m])
        std_bim   = np.std(resultados_bim[m])
        print(f"{m:<12}   {avg_bim:>10.2f} ±{std_bim:>5.2f}")
    print(f"\nPre-cálculos finalizados en {tiempo_precalc:.2f} segundos.")
