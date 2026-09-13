import numpy as np
from collections import deque


def balance_problem(cost_mat, sup_vec, dem_vec):
    tot_sup = np.sum(sup_vec)
    tot_dem = np.sum(dem_vec)

    if tot_sup == tot_dem:
        return cost_mat, sup_vec, dem_vec

    if tot_sup > tot_dem:
        diff = tot_sup - tot_dem
        dum_col = np.zeros((len(sup_vec), 1))
        cost_mat = np.hstack((cost_mat, dum_col))
        dem_vec = np.append(dem_vec, diff)
    else:
        diff = tot_dem - tot_sup
        dum_row = np.zeros((1, len(dem_vec)))
        cost_mat = np.vstack((cost_mat, dum_row))
        sup_vec = np.append(sup_vec, diff)

    return cost_mat, sup_vec, dem_vec


def add_basis_cells(curr_basis, alloc_mat):
    r_cnt, c_cnt = alloc_mat.shape

    while len(curr_basis) < r_cnt + c_cnt - 1:
        parent = list(range(r_cnt + c_cnt))

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a_idx, b_idx):
            root_a = find(a_idx)
            root_b = find(b_idx)
            if root_a != root_b:
                parent[root_b] = root_a
                return True
            return False

        for r_i, c_j in curr_basis:
            union(r_i, r_cnt + c_j)

        is_added = False
        for r_i in range(r_cnt):
            for c_j in range(c_cnt):
                if (r_i, c_j) not in curr_basis:
                    if find(r_i) != find(r_cnt + c_j):
                        curr_basis.add((r_i, c_j))
                        is_added = True
                        break
            if is_added:
                break

        if not is_added:
            raise ValueError("could not form a valid basis.")


def vam(cost_mat, sup_vec, dem_vec):
    sup_copy = sup_vec.copy().astype(float)
    dem_copy = dem_vec.copy().astype(float)

    n_r = len(sup_copy)
    n_c = len(dem_copy)

    alloc = np.zeros((n_r, n_c))
    basis_set = set()

    act_rows = [True] * n_r
    act_cols = [True] * n_c

    while any(act_rows) and any(act_cols):
        r_penalties = [None] * n_r
        c_penalties = [None] * n_c

        for i in range(n_r):
            if not act_rows[i]:
                continue
            vals = [cost_mat[i][j] for j in range(n_c) if act_cols[j]]
            vals.sort()
            if len(vals) >= 2:
                r_penalties[i] = vals[1] - vals[0]
            elif len(vals) == 1:
                r_penalties[i] = vals[0]

        for j in range(n_c):
            if not act_cols[j]:
                continue
            vals = [cost_mat[i][j] for i in range(n_r) if act_rows[i]]
            vals.sort()
            if len(vals) >= 2:
                c_penalties[j] = vals[1] - vals[0]
            elif len(vals) == 1:
                c_penalties[j] = vals[0]

        max_p = -1
        sel_r = -1
        sel_c = -1

        for i in range(n_r):
            if act_rows[i] and r_penalties[i] is not None:
                if r_penalties[i] > max_p:
                    max_p = r_penalties[i]
                    sel_r = i
                    sel_c = -1

        for j in range(n_c):
            if act_cols[j] and c_penalties[j] is not None:
                if c_penalties[j] > max_p:
                    max_p = c_penalties[j]
                    sel_c = j
                    sel_r = -1

        if sel_r != -1:
            i_picked = sel_r
            j_picked = min([j for j in range(n_c) if act_cols[j]], key=lambda j: cost_mat[i_picked][j])
        else:
            j_picked = sel_c
            i_picked = min([i for i in range(n_r) if act_rows[i]], key=lambda i: cost_mat[i][j_picked])

        allocated_qty = min(sup_copy[i_picked], dem_copy[j_picked])

        alloc[i_picked][j_picked] = allocated_qty
        basis_set.add((i_picked, j_picked))

        sup_copy[i_picked] -= allocated_qty
        dem_copy[j_picked] -= allocated_qty

        if sup_copy[i_picked] == 0 and dem_copy[j_picked] == 0:
            act_rows[i_picked] = False
            act_cols[j_picked] = False
        elif sup_copy[i_picked] == 0:
            act_rows[i_picked] = False
        elif dem_copy[j_picked] == 0:
            act_cols[j_picked] = False

    add_basis_cells(basis_set, alloc)
    return alloc, basis_set


def find_cycle(basis_set, enter_cell, n_r, n_c):
    r_start, c_start = enter_cell
    adj_graph = [[] for _ in range(n_r + n_c)]

    for r_idx, c_idx in basis_set:
        adj_graph[r_idx].append(n_r + c_idx)
        adj_graph[n_r + c_idx].append(r_idx)

    s_node = r_start
    t_node = n_r + c_start

    bfs_queue = deque([s_node])
    parents_map = {s_node: None}

    while bfs_queue:
        curr_node = bfs_queue.popleft()
        if curr_node == t_node:
            break

        for nxt in adj_graph[curr_node]:
            if nxt not in parents_map:
                parents_map[nxt] = curr_node
                bfs_queue.append(nxt)

    if t_node not in parents_map:
        return None

    path_nodes = []
    curr = t_node
    while curr is not None:
        path_nodes.append(curr)
        curr = parents_map[curr]

    path_nodes.reverse()

    loop_path = [enter_cell]
    for n1, n2 in zip(path_nodes, path_nodes[1:]):
        if n1 < n_r and n2 >= n_r:
            cell_pos = (n1, n2 - n_r)
        else:
            cell_pos = (n2, n1 - n_r)
        loop_path.append(cell_pos)

    return loop_path


def modi(cost_mat, alloc_mat, basis_set):
    n_r, n_c = alloc_mat.shape

    while True:
        u_vals = [None] * n_r
        v_vals = [None] * n_c

        u_vals[0] = 0.0
        updated = True

        while updated:
            updated = False
            for r_i, c_j in basis_set:
                if u_vals[r_i] is not None and v_vals[c_j] is None:
                    v_vals[c_j] = cost_mat[r_i][c_j] - u_vals[r_i]
                    updated = True
                elif v_vals[c_j] is not None and u_vals[r_i] is None:
                    u_vals[r_i] = cost_mat[r_i][c_j] - v_vals[c_j]
                    updated = True

        if any(val is None for val in u_vals) or any(val is None for val in v_vals):
            raise ValueError("invalid basic solution state.")

        entering_pos = None
        min_delta = 0.0
        alt_optima = False

        for r_i in range(n_r):
            for c_j in range(n_c):
                if (r_i, c_j) not in basis_set:
                    eval_cost = cost_mat[r_i][c_j] - u_vals[r_i] - v_vals[c_j]

                    if eval_cost < min_delta:
                        min_delta = eval_cost
                        entering_pos = (r_i, c_j)
                    elif eval_cost == 0:
                        alt_optima = True

        if entering_pos is None:
            final_c = np.sum(alloc_mat * cost_mat)
            return alloc_mat, final_c, alt_optima

        cycle_path = find_cycle(basis_set, entering_pos, n_r, n_c)
        if cycle_path is None:
            raise ValueError("could not trace improvement loop.")

        sub_cells = cycle_path[1::2]
        theta_val = min(alloc_mat[r_i][c_j] for r_i, c_j in sub_cells)

        for step_idx, (r_i, c_j) in enumerate(cycle_path):
            if step_idx % 2 == 0:
                alloc_mat[r_i][c_j] += theta_val
            else:
                alloc_mat[r_i][c_j] -= theta_val

        basis_set.add(entering_pos)

        for sub_pos in sub_cells:
            if np.isclose(alloc_mat[sub_pos], 0):
                basis_set.remove(sub_pos)
                break


supply = np.array([20, 30, 25])
demand = np.array([10, 20, 25, 20])

cost = np.array([
    [8, 6, 10, 9],
    [9, 7, 4, 2],
    [3, 5, 6, 4]
], dtype=float)


print("ambulance transportation problem")

print("problem:")
print("a city has 3 ambulance stations and 4 hospitals.")
print("the available ambulance trips at the stations are 20, 30 and 25.")
print("the required ambulance trips at the hospitals are 10, 20, 25 and 20.")

print("decision variables:")
print("xij = number of ambulance trips sent from station i to hospital j")

print("transportation cost matrix:")
print("             h1   h2   h3   h4")
print("station 1    ", cost[0])
print("station 2    ", cost[1])
print("station 3    ", cost[2])

print("objective function:")
print("minimize z =  8x11+ 6x12 +10x13 +  9x14 + "
      "9x21 +7x22 + 4x23+  2x24 + "
      "3x31 + 5x32+6x33 +  4x34")

print("subject to:")

print("x11+ x12 +  x13 + x14 =20")
print("x21 +  x22 + x23+ x24  = 30")
print("x31 + x32+  x33 +x34 =  25")

print("x11 +x21 + x31= 10")
print("x12+  x22 + x32 =  20")
print("x13 + x23+  x33=25")
print("x14+ x24 + x34  = 20")

print("xij >=0")


cost, supply, demand = balance_problem(
    cost,
    supply,
    demand
)


alloc_sol, basis_cells = vam(
    cost,
    supply,
    demand
)


print("initial basic feasible solution using vam:")
print("             h1   h2   h3   h4")

for i in range(len(supply)):
    print(
        "station", i + 1,
        alloc_sol[i].astype(int)
    )


init_cost = np.sum(alloc_sol * cost)

print("initial transportation cost = ", int(init_cost))


alloc_sol, opt_cost, has_mult = modi(
    cost,
    alloc_sol,
    basis_cells
)


print("optimal solution using modi:")
print("             h1   h2   h3   h4")

for i in range(len(supply)):
    print(
        "station", i + 1,
        alloc_sol[i].astype(int)
    )


print("minimum transportation cost = ", int(opt_cost))


if has_mult:
    print("multiple optimal solutions exist.")

print("the above allocation gives the minimum transportation cost.")