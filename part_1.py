import numpy as np

big_m = 1e6

def standardize(a_mat, b_vec, obj_coeffs, constraing_signs):
    a = np.array(a_mat, dtype=float)
    b = np.array(b_vec, dtype=float)
    c = list(map(float, obj_coeffs))

    n_rows = len(a)
    var_namse = [f"x{i + 1}" for i in range(len(a[0]))]

    basis_vars = []
    artif_indices = []

    s_idx = 0
    a_idx = 0

    for i in range(n_rows):
        sign = constraing_signs[i]
        
        if sign == "<=":
            col = np.zeros((n_rows, 1))
            col[i, 0] = 1.0
            a = np.hstack((a, col))

            s_idx += 1
            var_namse.append(f"s{s_idx}")
            c.append(0.0)
            basis_vars.append(len(var_namse) - 1)

        elif sign == ">=":
            col_surplus = np.zeros((n_rows, 1))
            col_surplus[i, 0] = -1.0
            a = np.hstack((a, col_surplus))
            s_idx += 1
            var_namse.append(f"s{s_idx}")
            c.append(0.0)

            col_art = np.zeros((n_rows, 1))
            col_art[i, 0] = 1.0
            a = np.hstack((a, col_art))
            a_idx += 1
            var_namse.append(f"a{a_idx}")
            c.append(-big_m)

            artif_indices.append(len(var_namse) - 1)
            basis_vars.append(len(var_namse) - 1)

        elif sign == "=":
            col_art = np.zeros((n_rows, 1))
            col_art[i, 0] = 1.0
            a = np.hstack((a, col_art))

            a_idx += 1
            var_namse.append(f"a{a_idx}")
            c.append(-big_m)

            artif_indices.append(len(var_namse) - 1)
            basis_vars.append(len(var_namse) - 1)

        else:
            raise ValueError(f"unknown constarint sign: {sign}")

    return a, b, np.array(c), var_namse, basis_vars, artif_indices


def print_standard_form(a_mat, b_vec, c_vec, var_namse):
    print("standerd form")
    for i in range(len(b_vec)):
        row_str = ""
        for j in range(len(var_namse)):
            coeff = a_mat[i][j]
            if abs(coeff) < 1e-10:
                continue

            if row_str != "":
                row_str += " + " if coeff > 0 else " - "
            elif coeff < 0:
                row_str += "-"

            val = abs(coeff)
            if val != 1:
                row_str += str(int(val))
            row_str += var_namse[j]

        print(row_str, "=", int(b_vec[i]))

    print("big-m objcetive:")
    obj_str = "z = "
    for j in range(len(var_namse)):
        if c_vec[j] == 0:
            continue

        if obj_str != "z = ":
            obj_str += " + " if c_vec[j] > 0 else " - "
        elif c_vec[j] < 0:
            obj_str += "-"

        val = abs(c_vec[j])
        if val == big_m:
            obj_str += "m"
        else:
            obj_str += str(int(val))
        obj_str += var_namse[j]

    print(obj_str)


def big_m_simplex(a_mat, b_vec, obj_coeffs, constraing_signs):
    num_orig_vars = len(obj_coeffs)
    a, b, c, var_namse, curr_basis, artif_idx = standardize(
        a_mat, b_vec, obj_coeffs, constraing_signs
    )

    m, n = len(b), len(c)
    tableau = np.hstack((a, b.reshape(-1, 1)))

    for step in range(100):
        c_basis = c[curr_basis]
        zj = c_basis @ tableau[:, :n]
        reduced_costs = c - zj

        print("itration", step)
        print("baisis:", [var_namse[k] for k in curr_basis])
        print("cj - zj:", np.round(reduced_costs, 3))

        candidats = [j for j in range(n) if j not in curr_basis and reduced_costs[j] > 1e-9]

        if not candidats:
            sol = np.zeros(n)
            for r in range(m):
                sol[curr_basis[r]] = tableau[r, -1]

            for art in artif_idx:
                if sol[art] > 1e-7:
                    return None, var_namse, "infeasible"

            is_multiple = any(j not in curr_basis and abs(reduced_costs[j]) <= 1e-8 for j in range(num_orig_vars))
            if is_multiple:
                return sol, var_namse, "multiple"

            return sol, var_namse, "optimal"

        piv_col = max(candidats, key=lambda j: reduced_costs[j])

        ratios = []
        for r in range(m):
            if tableau[r, piv_col] > 1e-10:
                ratios.append((tableau[r, -1] / tableau[r, piv_col], r))

        if not ratios:
            return None, var_namse, "unbounded"

        _, piv_row = min(ratios, key=lambda x: (x[0], x[1]))

        tableau[piv_row] /= tableau[piv_row, piv_col]
        for r in range(m):
            if r != piv_row:
                tableau[r] -= tableau[r, piv_col] * tableau[piv_row]

        curr_basis[piv_row] = piv_col

    raise RuntimeError("maximum number of itrations exceeded.")


variable_names = ["x1", "x2", "x3", "x4"]
c = [40, 35, 25, 30]

a = [
    [2, 1, 1, 3],
    [1, 2, 3, 1],
    [1, 1, 1, 1],
    [3, 1, 2, 1]
]

signs = ["<=", ">=", "=", "<="]
b = [40, 30, 12, 25]
prob_type = 1

print("advertisng media selection problm")
print("problm:")
print("a company wants to select advertisments from tv, radio,")
print("newspapr and online media.")

print("decison variables:")
print("x1 = number of tv advertismnts")
print("x2 = number of radio advertisments")
print("x3 = number of newspapr advertisments")
print("x4 = number of online advertisments")

print("objctive function:")
print("maxmize z = 40x1 + 35x2 + 25x3 + 30x4")

print("subjet to:")
print("2x1 + x2 + x3 + 3x4 <= 40")
print("x1 + 2x2 + 3x3 + x4 >= 30")
print("x1 + x2 + x3 + x4 = 12")
print("3x1 + x2 + 2x3 + x4 <= 25")
print("x1, x2, x3, x4 >= 0")


if prob_type == 2:
    c = [-val for val in c]


std_a, std_b, std_c, names, _, _ = standardize(a, b, c, signs)
print_standard_form(std_a, std_b, std_c, names)


sol, names, stat = big_m_simplex(a, b, c, signs)


print("fianl result")
if stat in ("optimal", "multiple"):
    if stat == "multiple":
        print("multple optimal solutins exist.")
    else:
        print("optiaml solution:")

    for i in range(len(variable_names)):
        print(variable_names[i], "=", round(sol[i], 4))

    z_val = np.dot(c, sol[:len(c)])
    if prob_type == 2:
        z_val = -z_val
        print("minimun z =", round(z_val, 4))
    else:
        print("maxmum z =", round(z_val, 4))

elif stat == "infeasible":
    print("the problm is infeasable.")
elif stat == "unbounded":
    print("the problm is unboudned.")