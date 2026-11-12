"""联合多分位分位回归：一个 LP 同时拟合全部分位并强制域内不交叉。

模型（标准化协变量 z，含截距）：q_k(z) = β_{k0} + Σ_{j>=1} β_{kj} z_j。

联合目标（i 只遍历训练窗完整行，w_i 为正权重）：
    Σ_k Σ_i w_i · pinball_{τ_k}(y_i - q_k(z_i))
      + λ · Σ_k Σ_{j>=1} |β_{kj}|                 （截距不惩罚）

不交叉在调用者给出的凸包**每个公开顶点**上约束相邻分位差非负：
    q_{k+1}(v) - q_k(v) >= 0   对全部顶点 v；
线性预测的非负性沿凸组合推广，保证覆盖整个凸包，且仅覆盖该凸包。

LP 等价式（所有变量非负）：
    残差 r = y - q = u - v，u,v>=0，
    pinball_τ(r) = τ u + (1-τ) v（最优解互补，u·v=0）；
    自由系数 β = bp - bn，bp,bn>=0，斜率惩罚变为 λ(bp+bn)，截距不惩罚。
解可能不唯一：输出一组声明误差内满足目标与约束的合法系数并附最优性
证据，不编造唯一系数。
"""
import math
from dataclasses import dataclass, field

from .lp import linprog


def pinball(tau, residual):
    """残差 r=y-预测：r>=0 时 τr，否则 (τ-1)r。逐行可独立重算。"""
    return tau * residual if residual >= 0.0 else (tau - 1.0) * residual


def design_row(z):
    """标准化协变量前加截距列 1。"""
    return [1.0] + [float(v) for v in z]


@dataclass
class JointFit:
    status: str
    lamb: float
    taus: list[float]
    coefficients: list[list[float]] | None      # [k][p]，p=1+d，首列为截距
    objective: float | None                      # 惩罚后联合目标（独立重算）
    pinball_total: float | None                 # 未惩罚 pinball 总和
    penalty: float | None
    solver: dict = field(default_factory=dict)
    vertex_gaps: dict = field(default_factory=dict)  # 相邻对 -> 各顶点差值
    min_vertex_gap: float | None = None
    rank: int | None = None
    rank_deficient: bool | None = None
    constant_covariates: list[bool] = field(default_factory=list)


def _matrix_rank(rows, tol=1e-12):
    """高斯消元求秩，rows 为行向量列表（含截距的设计矩阵）。"""
    mat = [list(r) for r in rows]
    if not mat:
        return 0
    m, n = len(mat), len(mat[0])
    rank = 0
    scale = max([1.0] + [abs(v) for row in mat for v in row])
    eps = tol * scale
    for col in range(n):
        pivot = next((h for h in range(rank, m) if abs(mat[h][col]) > eps), None)
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        pe = mat[rank][col]
        for h in range(rank + 1, m):
            factor = mat[h][col] / pe
            if factor == 0.0:
                continue
            for j in range(col, n):
                mat[h][j] -= factor * mat[rank][j]
        rank += 1
        if rank == m:
            break
    return rank


def fit_joint(taus, z_rows, y_rows, weights, vertices_z, lamb,
              constant_covariates, tolerance=1e-8, max_iterations=10000):
    """对完整训练行联合拟合。vertices_z 为同一标准化下的公开顶点。"""
    taus = [float(t) for t in taus]
    k_n = len(taus)
    n = len(z_rows)
    if not n:
        raise ValueError('没有完整训练行')
    d = len(z_rows[0])
    p = d + 1
    if any(len(z) != d for z in z_rows):
        raise ValueError('协变量维度不一致')
    Z = [design_row(z) for z in z_rows]
    V = [design_row(v) for v in vertices_z]

    # 变量布局（全部非负）：
    # u(K*n), v(K*n), bp(K*p), bn(K*p)；β=bp-bn。
    iu = lambda k, i: k * n + i
    iv = lambda k, i: k_n * n + k * n + i
    ip = lambda k, j: 2 * k_n * n + k * p + j
    inn = lambda k, j: 2 * k_n * n + k_n * p + k * p + j
    nvar = 2 * k_n * n + 2 * k_n * p

    c = [0.0] * nvar
    for k, tau in enumerate(taus):
        for i, w in enumerate(weights):
            c[iu(k, i)] = float(w) * tau
            c[iv(k, i)] = float(w) * (1.0 - tau)
        for j in range(1, p):  # 截距 j=0 不惩罚
            c[ip(k, j)] = float(lamb)
            c[inn(k, j)] = float(lamb)

    A_eq, b_eq = [], []
    # u - v + Σ Z_ij (bp - bn) = y
    for k in range(k_n):
        for i in range(n):
            row = [0.0] * nvar
            row[iu(k, i)] = 1.0
            row[iv(k, i)] = -1.0
            for j in range(p):
                row[ip(k, j)] = Z[i][j]
                row[inn(k, j)] = -Z[i][j]
            A_eq.append(row)
            b_eq.append(float(y_rows[i]))

    A_ub, b_ub = [], []
    # 顶点不交叉：q_k(v) - q_{k+1}(v) <= 0
    for k in range(k_n - 1):
        for vrow in V:
            row = [0.0] * nvar
            for j in range(p):
                row[ip(k, j)] = vrow[j]
                row[inn(k, j)] = -vrow[j]
                row[ip(k + 1, j)] = -vrow[j]
                row[inn(k + 1, j)] = vrow[j]
            A_ub.append(row)
            b_ub.append(0.0)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  tolerance=tolerance, max_iterations=max_iterations)
    if res.status != 'optimal':
        return JointFit(status=res.status, lamb=float(lamb), taus=taus,
                        coefficients=None, objective=None,
                        pinball_total=None, penalty=None,
                        solver={'status': res.status,
                                'iterations': res.iterations},
                        constant_covariates=list(constant_covariates))

    beta = [[res.x[ip(k, j)] - res.x[inn(k, j)] for j in range(p)]
            for k in range(k_n)]

    # ---- 独立复核：不信任 LP 内部目标值 ----
    pinball_total = 0.0
    for k, tau in enumerate(taus):
        for i in range(n):
            pred = sum(beta[k][j] * Z[i][j] for j in range(p))
            pinball_total += float(weights[i]) * pinball(
                tau, float(y_rows[i]) - pred)
    penalty = float(lamb) * sum(abs(beta[k][j])
                                for k in range(k_n) for j in range(1, p))
    objective = pinball_total + penalty

    vertex_gaps = {}
    min_gap = math.inf
    for k in range(k_n - 1):
        gaps = []
        for vrow in V:
            gap = sum((beta[k + 1][j] - beta[k][j]) * vrow[j] for j in range(p))
            gaps.append(gap)
            min_gap = min(min_gap, gap)
        vertex_gaps[f'{taus[k]}->{taus[k + 1]}'] = gaps

    rank = _matrix_rank(Z)
    rank_deficient = rank < min(n, p)

    return JointFit(
        status='optimal', lamb=float(lamb), taus=taus, coefficients=beta,
        objective=objective, pinball_total=pinball_total, penalty=penalty,
        solver={'status': res.status, 'lp_objective': res.objective,
                'objective_recomputed': objective,
                'objective_abs_diff': abs(objective - res.objective),
                'primal_residual': res.primal_residual,
                'dual_residual': res.dual_residual,
                'duality_gap': res.duality_gap,
                'iterations': res.iterations,
                'dropped_rows': res.dropped_rows},
        vertex_gaps=vertex_gaps, min_vertex_gap=min_gap,
        rank=rank, rank_deficient=rank_deficient,
        constant_covariates=list(constant_covariates))


def predict(fit, z):
    """用冻结系数给出各分位线性预测；调用者须先完成凸包成员判定。"""
    if fit.coefficients is None:
        raise ValueError('拟合未达最优，无法预测')
    zfull = design_row(z)
    return [sum(fit.coefficients[k][j] * zfull[j]
                for j in range(len(zfull))) for k in range(len(fit.taus))]
