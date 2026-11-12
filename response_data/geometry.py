"""有界凸包成员判定，全部在原协变量单位进行。

成员见证为非负重心权重 λ：λ>=0、Σλ=1、V'λ=p。
可行性由 ``response_data.lp`` 的两阶段单纯形判定，不依赖训练样本点——
不交叉保证的适用范围只由调用者给出的顶点凸包界定。
"""
from dataclasses import dataclass

from .lp import linprog


@dataclass
class HullMembership:
    inside: bool
    witness: list[float]          # 与顶点同序的重心权重（域外全为 0）
    equality_residual: float      # V'λ=p 与 Σλ=1 的最大绝对偏差
    status: str                   # LP 状态，见证失败不得声称域内


def _barycentric(point, vertices, tolerance, max_iterations):
    m = len(vertices)
    d = len(point)
    if m == 0 or any(len(v) != d for v in vertices):
        raise ValueError('顶点与点的维度不一致')
    A_eq = [[1.0] * m]
    b_eq = [1.0]
    for j in range(d):
        A_eq.append([float(vertices[i][j]) for i in range(m)])
        b_eq.append(float(point[j]))
    res = linprog([0.0] * m, A_eq=A_eq, b_eq=b_eq,
                  tolerance=tolerance, max_iterations=max_iterations)
    return res


def membership(point, vertices, tolerance=1e-9, max_iterations=10000):
    """返回凸包成员判定与重心见证；LP 未成功时按域外处理并保留状态。"""
    res = _barycentric(point, vertices, tolerance, max_iterations)
    if res.status != 'optimal' or res.x is None:
        return HullMembership(False, [0.0] * len(vertices), float('nan'),
                              res.status)
    witness = [max(0.0, lam) for lam in res.x]
    residual = max(
        abs(sum(witness) - 1.0),
        *(abs(sum(vertices[i][j] * witness[i] for i in range(len(vertices)))
              - point[j]) for j in range(len(point))))
    inside = residual <= max(tolerance * 100.0, 1e-9) * max(
        1.0, *(abs(v) for vert in vertices for v in vert))
    return HullMembership(inside, witness, residual, res.status)
