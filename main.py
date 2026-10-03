import networkx as nx
import igraph as ig
import random
import numpy as np
import pandas as pd
import time
from typing import List

class DCSA_BIM:
    def __init__(self, G: nx.Graph, n: int, B: float, alpha: float, beta: float, g_max: int, fl: float, AP: float, 
                 costs: dict, efficiencies: dict, constraint_dict: dict, p: float = 0.01):
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
        
        self.lie_cache = {} # Memoria caché para no recalcular LIE

    def _get_cost(self, S: List[int]) -> float:
        """Calcula el costo total de un conjunto semilla"""
        return sum(self.costs[n] for n in S)

    def calculate_LIE(self, S: List[int]) -> float:
        """Estimación de Influencia Local (LIE) de 2 saltos con Caché"""
        if not S:
            return 0.0

        # Convertir la lista a un frozenset (inmutable) para usarlo como llave
        s_key = frozenset(S)
        if s_key in self.lie_cache:
            return self.lie_cache[s_key] # Si ya se calculó antes, retornar el valor al instante

        S_set = set(S)
        N1_S = set()
        for u in S:
            N1_S.update(self.G.neighbors(u))
        N1_S = N1_S - S_set

        N2_S = set()
        for u in N1_S:
            N2_S.update(self.G.neighbors(u))
        N2_S = N2_S - S_set - N1_S

        lie_value = float(len(S))

        inf_1 = 0.0
        for v in N1_S:
            neighbors_in_S = sum(1 for u in self.G.neighbors(v) if u in S_set)
            inf_1 += (1 - (1 - self.p) ** neighbors_in_S)

        lie_value += inf_1

        if len(N1_S) > 0:
            sum_p_d = sum(self.p * self.G.degree(u) for u in N2_S)
            lie_value += (inf_1 / len(N1_S)) * sum_p_d

        # Guardar el resultado en el caché antes de retornar
        self.lie_cache[s_key] = lie_value
        return lie_value

    def generate_candidates(self, alpha_metric: float = 0.5, beta_cand: float = 3.0) -> List[int]:
        """Genera candidatos usando K-shell y Structural Holes"""
        k_shell = nx.core_number(self.G)
        sum_ks = sum(k_shell.values())
        """
        try:
            struct_holes = nx.constraint(self.G)
        except:
            struct_holes = {node: 1.0/self.G.degree(node) if self.G.degree(node)>0 else 1.0 for node in self.nodes}
        """
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
        Verifica el presupuesto B. Si excede, elimina secuencialmente
        los nodos de menor eficiencia e(v) hasta cumplir la restricción
        """
        S_set = set(S)

        while self._get_cost(list(S_set)) > self.B:
            # Seleccionar el nodo con la peor eficiencia
            worst_node = min(S_set, key=lambda node: self.efficiencies[node])
            S_set.remove(worst_node)

            # Intentar rellenar con un nodo aleatorio que encaje en el saldo
            remaining_budget = self.B - self._get_cost(list(S_set))
            if remaining_budget >= 1.0:
                valid_candidates = [n for n in self.nodes if n not in S_set and self.costs[n] <= remaining_budget]
                if valid_candidates:
                    new_node = random.choice(valid_candidates)
                    S_set.add(new_node)

        return list(S_set)

    def initialization(self) -> List[List[int]]:
        """Inicialización priorizando nodos en orden descendente según su eficiencia e(v)"""
        sorted_by_eff = sorted(self.nodes, key=lambda x: self.efficiencies[x], reverse=True)

        base_S = []
        current_cost = 0.0
        for node in sorted_by_eff:
            if current_cost + self.costs[node] <= self.B:
                base_S.append(node)
                current_cost += self.costs[node]

        X = []
        for i in range(self.n):
            xi = list(base_S)
            for j in range(len(xi)):
                if random.random() > 0.5:
                    node_to_remove = xi[j]
                    cost_without = self._get_cost(xi) - self.costs[node_to_remove]
                    valid_candidates = [n for n in self.nodes if n not in xi and cost_without + self.costs[n] <= self.B]
                    if valid_candidates:
                        xi[j] = random.choice(valid_candidates)
            X.append(xi)
        return X

    def update_position(self, x_i: List[int], m_j: List[int], r_i: float) -> List[int]:
        """Actualiza la posición y repara la solución"""
        x_new = list(x_i)
        m_j_set = set(m_j)

        for j in range(len(x_new)):
            intersection_val = 0 if x_new[j] in m_j_set else 1
            factor = r_i * self.fl * intersection_val
            H = 1 if factor >= 1 else 0

            if H == 1:
                while True:
                    new_node = random.choice(self.nodes)
                    if new_node not in x_new:
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
        current_lie = self.calculate_LIE(x_best)
        current_cost = self._get_cost(x_best)

        # Ordenar por eficiencia ascendente
        sorted_x = sorted(x_best, key=lambda n: self.efficiencies[n])

        for node in sorted_x:
            neighbors = list(self.G.neighbors(node))

            # Búsqueda Local Truncada.
            # Si el nodo tiene demasiados vecinos, evalúa solo a los 15 más eficientes
            if len(neighbors) > 15:
                neighbors = sorted(neighbors, key=lambda n: self.efficiencies[n], reverse=True)[:15]

            for neighbor in neighbors:
                if neighbor not in x_best:
                    cost_diff = self.costs[neighbor] - self.costs[node]

                    if current_cost + cost_diff <= self.B:
                        x_temp = list(x_best)
                        idx = x_temp.index(node)
                        x_temp[idx] = neighbor

                        new_lie = self.calculate_LIE(x_temp)
                        if new_lie > current_lie:
                            x_best = x_temp
                            current_lie = new_lie
                            current_cost += cost_diff
                            node = neighbor # Se actualiza para que las siguientes comparaciones sean contra el nuevo
        return x_best

    def optimize(self):
        """Ejecuta el ciclo principal del DCSA."""
        X = self.initialization()
        M = [list(xi) for xi in X]

        LIE_values = [self.calculate_LIE(xi) for xi in X]
        best_idx = np.argmax(LIE_values)
        x_star = list(X[best_idx])

        Candidates = self.generate_candidates()

        for iter_num in range(self.g_max):
            for i in range(self.n):
                j = random.randint(0, self.n - 1)
                r_i = random.random()
                r_j = random.random()

                if r_j >= self.AP:
                    x_new = self.update_position(X[i], M[j], r_i)
                else:
                    x_new = self.random_walk(Candidates)

                if self.calculate_LIE(x_new) > self.calculate_LIE(X[i]):
                    X[i] = list(x_new)

                if self.calculate_LIE(X[i]) > self.calculate_LIE(M[i]):
                    M[i] = list(X[i])

            LIE_values = [self.calculate_LIE(xi) for xi in X]
            best_idx = np.argmax(LIE_values)
            current_best = list(X[best_idx])

            if self.calculate_LIE(current_best) > self.calculate_LIE(x_star):
                x_star = current_best

            x_star_prime = self.local_search(x_star)
            if self.calculate_LIE(x_star_prime) > self.calculate_LIE(x_star):
                x_star = x_star_prime

            print(f"Iteración {iter_num + 1}/{self.g_max} | Mejor LIE: {self.calculate_LIE(x_star):.4f} | Nodos: {len(x_star)} | Costo: {self._get_cost(x_star):.2f}/{self.B:.2f}")

        return x_star

def calcular_spread_monte_carlo(G, semillas, p=0.01, iteraciones=10000):
    """Ejecuta el modelo IC múltiples veces para calcular el spread esperado (promedio)"""
    spread_total = 0
    for _ in range(iteraciones):
        nodos_activos = set(semillas)
        nuevos_activos = set(semillas)

        while nuevos_activos:
            proximos_activos = set()
            for u in nuevos_activos:
                for v in G.neighbors(u):
                    if v not in nodos_activos and v not in proximos_activos:
                        if random.random() < p:
                            proximos_activos.add(v)
            nuevos_activos = proximos_activos
            nodos_activos.update(proximos_activos)

        spread_total += len(nodos_activos)

    return spread_total / iteraciones

if __name__ == "__main__":

    N_RUNS = 10

    # Cargar la red
    formato = input("Formato archivo (txt/csv): ")
    if formato == "txt":
        print("Cargando la red desde txt...")
        G = nx.read_edgelist("Redes/nombre_red.txt", comments='#', nodetype=int, create_using=nx.Graph())
    else:
        print("Cargando la red desde csv...")
        df = pd.read_csv("Redes/nombre_red.csv")
        G = nx.from_pandas_edgelist(df, source='nodo_1', target='nodo_2', create_using=nx.Graph())

    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f"Red cargada: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

    # =====================================================================
    # BLOQUE DE PRE-CÁLCULOS (SE EJECUTA UNA SOLA VEZ)
    # =====================================================================
    print("\nIniciando pre-cálculo de métricas estructurales de la red...")
    start_precalc = time.time()

    # 1. igraph y Constraint (Structural Holes)
    print("- Construyendo igraph y calculando Constraint...")
    H = ig.Graph.from_networkx(G)
    constraints_list = H.constraint()
    constraint_dict = dict(zip(H.vs["_nx_name"], constraints_list))

    # 2. Betweenness Centrality
    print("- Calculando Betweenness Centrality...")
    bc_values = H.betweenness(directed=False, normalized=True, vertices=None, cutoff=-1)
    bc = dict(zip(H.vs["_nx_name"], bc_values))
    bc_min, bc_max = min(bc.values()), max(bc.values())

    # 3. Grados, Costos y Eficiencias
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

        # --- DCSA-BIM ---
        # Pasar diccionarios precalculados a la clase
        dcsa_bim = DCSA_BIM(
            G=G, n=20, B=50, alpha=alpha_costo, beta=beta_eficiencia, g_max=100, fl=1.5, AP=0.1,
            costs=costs_dict, 
            efficiencies=efficiencies_dict, 
            constraint_dict=constraint_dict
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