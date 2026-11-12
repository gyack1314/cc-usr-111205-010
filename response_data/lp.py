"""两阶段单纯形求解标准形线性规划。

求解 min c'x，约束 A_ub x <= b_ub、A_eq x = b_eq，x >= 0。
返回原始解、对偶变量、原始/对偶可行性残差与对偶间隙，并显式区分
optimal / infeasible / unbounded / iter_limit——未收敛结果不得称为最优。
仅依赖 Python 标准库。
"""
from dataclasses import dataclass, field


@dataclass
class LPResult:
    status: str
    x: list[float] | None = None
    dual: list[float] | None = None
    objective: float | None = None
    reduced_costs: list[float] | None = None
    primal_residual: float | None = None
    dual_residual: float | None = None
    duality_gap: float | None = None
    iterations: int = 0
    dropped_rows: list[int] = field(default_factory=list)


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
            tolerance=1e-9, max_iterations=10000):
    c = [float(v) for v in c]
    n = len(c)
    A_ub = [list(map(float, row)) for row in (A_ub or [])]
    b_ub = [float(v) for v in (b_ub or [])]
    A_eq = [list(map(float, row)) for row in (A_eq or [])]
    b_eq = [float(v) for v in (b_eq or [])]
    if A_ub and any(len(row) != n for row in A_ub):
        raise ValueError('A_ub 列数与目标变量数不一致')
    if A_eq and any(len(row) != n for row in A_eq):
        raise ValueError('A_eq 列数与目标变量数不一致')
    if len(A_ub) != len(b_ub) or len(A_eq) != len(b_eq):
        raise ValueError('约束矩阵与右端长度不一致')

    # 无约束情形：x=0 可行，存在负费用方向则无界。
    if not A_ub and not A_eq:
        if any(v < -tolerance for v in c):
            return LPResult('unbounded')
        x = [0.0] * n
        return LPResult('optimal', x=x, dual=[], objective=0.0,
                        reduced_costs=list(c), primal_residual=0.0,
                        dual_residual=max([0.0] + [max(0.0, -v) for v in c]),
                        duality_gap=0.0)

    # 列布局：真实变量 [0,n)，松弛变量，人工变量（等式行 + 负rhs不等式行）。
    n_art = len(A_eq) + sum(1 for b in b_ub if b < 0)
    width = n + len(A_ub) + n_art
    slack_cols: list[int | None] = []   # 每个 ub 行的松弛列
    art_cols: list[int | None] = []     # 每个约束行的人工列（无则 None）
    row_flip: list[float] = []          # 该行是否整体乘过 -1
    tab: list[list[float]] = []
    rhs: list[float] = []
    basis: list[int] = []
    identity_col: list[int] = []
    art_set: set[int] = set()
    art_count = 0

    next_slack = n
    for a, b in zip(A_ub, b_ub):
        row = [0.0] * width
        if b >= 0:
            for j, v in enumerate(a):
                row[j] = v
            col = next_slack
            row[col] = 1.0
            basis.append(col)
            slack_cols.append(col)
            art_cols.append(None)
            row_flip.append(1.0)
        else:
            # b<0：整行乘 -1 后加人工变量。
            for j, v in enumerate(a):
                row[j] = -v
            col = next_slack
            row[col] = -1.0  # 松弛列保留，供对偶使用
            art = n + len(A_ub) + art_count
            row[art] = 1.0
            art_cols.append(art)
            art_set.add(art)
            art_count += 1
            basis.append(art)
            slack_cols.append(col)
            row_flip.append(-1.0)
        identity_col.append(basis[-1])
        tab.append(row)
        rhs.append(abs(b))
        next_slack += 1

    for a, b in zip(A_eq, b_eq):
        row = [0.0] * width
        sign = 1.0 if b >= 0 else -1.0
        for j, v in enumerate(a):
            row[j] = sign * v
        art = n + len(A_ub) + art_count
        row[art] = 1.0
        art_cols.append(art)
        art_set.add(art)
        art_count += 1
        row_flip.append(sign)
        identity_col.append(art)
        basis.append(art)
        tab.append(row)
        rhs.append(sign * b)

    m = len(tab)
    scale_b = max([1.0] + [abs(v) for v in rhs])
    scale_c = max([1.0] + [abs(v) for v in c])
    eps_pri = tolerance * scale_b
    eps_dual = tolerance * scale_c

    def make_objective(cost):
        rows = [i for i, bcol in enumerate(basis) if bcol in art_set]
        r = [0.0] * width
        for j in range(width):
            r[j] = cost(j) - sum(tab[h][j] for h in rows)
        z = sum(rhs[h] for h in rows)
        return r, -z

    def do_pivot(r, r_rhs, enter):
        # Bland 比值检验，平局取基变量下标最小者；负系数行不限制步长，
        # 全部 <=0 即真实无界射线（配合 Bland 入基规则无需退化负枢轴）。
        leaving = None
        best = None
        for i in range(len(tab)):
            a = tab[i][enter]
            if a > eps_pri:
                ratio = rhs[i] / a
                if best is None or ratio < best - eps_pri or \
                        (abs(ratio - best) <= eps_pri and basis[i] < basis[leaving]):
                    best, leaving = ratio, i
        if leaving is None:
            return None
        pe = tab[leaving][enter]
        for j in range(width):
            tab[leaving][j] /= pe
        rhs[leaving] /= pe
        pivot_row = tab[leaving]
        pivot_rhs = rhs[leaving]
        for i in range(len(tab)):
            if i == leaving:
                continue
            factor = tab[i][enter]
            if factor == 0.0:
                continue
            target = tab[i]
            for j in range(width):
                target[j] -= factor * pivot_row[j]
            rhs[i] -= factor * pivot_rhs
        r_factor = r[enter]
        if r_factor != 0.0:
            for j in range(width):
                r[j] -= r_factor * pivot_row[j]
            r_rhs[0] -= r_factor * pivot_rhs
        basis[leaving] = enter
        return leaving

    iterations = 0

    def simplex_phase(r, r_rhs, phase):
        nonlocal iterations
        # 第二阶段人工列费用置 0 且禁止入基，防止真实变量被逐出可行基。
        forbidden = art_set if phase == 2 else set()
        while True:
            enter = None
            for j in range(width):
                if j in forbidden:
                    continue
                if r[j] < -eps_dual:
                    enter = j  # Bland：最小下标
                    break
            if enter is None:
                return 'optimal', r_rhs[0]
            if iterations >= max_iterations:
                return 'iter_limit', r_rhs[0]
            leaving = do_pivot(r, r_rhs, enter)
            if leaving is None:
                return 'unbounded', r_rhs[0]
            iterations += 1

    # ---- 第一阶段：最小化人工变量之和 ----
    d = [1.0 if j in art_set else 0.0 for j in range(width)]
    r, r_rhs = make_objective(lambda j: d[j])
    r_rhs = [r_rhs]
    status, r_rhs_value = simplex_phase(r, r_rhs, 1)
    if status == 'iter_limit':
        return LPResult('iter_limit', iterations=iterations)
    if status == 'unbounded' or -r_rhs_value > eps_pri:
        return LPResult('infeasible', iterations=iterations)

    # 人工变量仍在基中（值为 0）：换入非人工列，否则该行冗余删除。
    keep = list(range(len(tab)))
    for i in list(range(len(tab))):
        if basis[i] not in art_set:
            continue
        pivot_col = None
        for j in range(width):
            if j in art_set:
                continue
            if abs(tab[i][j]) > eps_pri:
                pivot_col = j
                break
        if pivot_col is None:
            keep.remove(i)
            continue
        pe = tab[i][pivot_col]
        for j in range(width):
            tab[i][j] /= pe
        rhs[i] /= pe
        for h in range(len(tab)):
            if h == i:
                continue
            factor = tab[h][pivot_col]
            if factor == 0.0:
                continue
            for j in range(width):
                tab[h][j] -= factor * tab[i][j]
            rhs[h] -= factor * rhs[i]
        r_factor = r[pivot_col]
        pivot_rhs_i = rhs[i]
        for j in range(width):
            r[j] -= r_factor * tab[i][j]
        r_rhs[0] -= r_factor * pivot_rhs_i
        basis[i] = pivot_col
    if len(keep) != len(tab):
        dropped = [i for i in range(m) if i not in keep]
        tab = [tab[i] for i in keep]
        rhs = [rhs[i] for i in keep]
        basis = [basis[i] for i in keep]
        identity_col = [identity_col[i] for i in keep]
        row_flip = [row_flip[i] for i in keep]
    else:
        dropped = []

    # ---- 第二阶段：真实目标，人工列费用置 0 保留在表中 ----
    # 表内目标行约定为 r_j = c_j - c_B B^-1 A_j，右端存 -z。
    r2 = [0.0] * width
    for j in range(n):
        r2[j] = c[j]
    for i, bcol in enumerate(basis):
        cb = c[bcol] if bcol < n else 0.0
        if cb == 0.0:
            continue
        for j in range(width):
            r2[j] -= cb * tab[i][j]
    z0 = sum((c[bcol] if bcol < n else 0.0) * rhs[i]
             for i, bcol in enumerate(basis))
    r_rhs2 = [-z0]
    status, r_rhs2_value = simplex_phase(r2, r_rhs2, 2)
    if status == 'iter_limit':
        return LPResult('iter_limit', iterations=iterations)
    if status == 'unbounded':
        return LPResult('unbounded', iterations=iterations)

    x = [0.0] * n
    for i, bcol in enumerate(basis):
        if bcol < n:
            x[bcol] = rhs[i]
    objective = -r_rhs2_value

    # 对偶变量：每行初始单位列的约化费用 r=-y_tableau；
    # 符号翻转过的行要翻回原约束方向（ub 对偶 <=0，eq 自由）；
    # 被删除的冗余行对偶取 0（该约束在原系统中不独立）。
    kept_dual = {}
    for k, row_index in enumerate(keep):
        kept_dual[row_index] = -r2[identity_col[k]] * row_flip[k]
    dual = [kept_dual.get(i, 0.0) for i in range(len(A_ub) + len(A_eq))]

    # 独立残差核算，不信任表内自报。
    primal_residual = 0.0
    for a, b in zip(A_ub, b_ub):
        violation = sum(a[j] * x[j] for j in range(n)) - b
        primal_residual = max(primal_residual, max(0.0, violation))
    for a, b in zip(A_eq, b_eq):
        primal_residual = max(
            primal_residual, abs(sum(a[j] * x[j] for j in range(n)) - b))
    dual_residual = max([0.0] + [max(0.0, -v) for v in r2[:n]])
    b_all = b_ub + b_eq
    dual_obj = sum(bi * yi for bi, yi in zip(b_all, dual))
    gap = abs(objective - dual_obj)

    return LPResult('optimal', x=x, dual=dual, objective=objective,
                    reduced_costs=r2[:n], primal_residual=primal_residual,
                    dual_residual=dual_residual, duality_gap=gap,
                    iterations=iterations, dropped_rows=dropped)
