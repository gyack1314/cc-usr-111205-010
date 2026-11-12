"""训练窗锁定的标准化与完整行切分。

中心为正权重加权均值，正 scale 为加权总体标准差；某维标准差为零时
scale 固定为 1 并登记 constant 诊断。中心/scale 只由调用时给定的训练
窗完整行决定，验证行、外期行与顶点只套用同一变换、不参与计算。
"""
import math
from dataclasses import dataclass, field


@dataclass
class Standardizer:
    centers: list[float]
    scales: list[float]
    constant: list[bool]
    weight_total: float
    complete_rows: int

    def transform(self, x):
        return [(float(x[j]) - self.centers[j]) / self.scales[j]
                for j in range(len(self.centers))]


def is_complete(row, d):
    return row.get('y') is not None and all(
        v is not None for v in row['x']) and len(row['x']) == d


def split_rows(rows, d):
    """完整行与缺字段行分开；缺失行不参与任何拟合。"""
    complete, incomplete = [], []
    for row in rows:
        (complete if is_complete(row, d) else incomplete).append(row)
    return complete, incomplete


def fit_standardizer(rows, d):
    """只用这些（训练窗）完整行锁定中心与 scale。"""
    complete, _ = split_rows(rows, d)
    if not complete:
        raise ValueError('训练窗没有可用于拟合的完整行')
    wsum = sum(float(r['weight']) for r in complete)
    centers = [sum(float(r['weight']) * float(r['x'][j]) for r in complete)
               / wsum for j in range(d)]
    scales, constant = [], []
    for j in range(d):
        var = sum(float(r['weight']) *
                  (float(r['x'][j]) - centers[j]) ** 2 for r in complete) / wsum
        std = math.sqrt(max(var, 0.0))
        if std == 0.0:
            scales.append(1.0)
            constant.append(True)
        else:
            scales.append(std)
            constant.append(False)
    return Standardizer(centers, scales, constant, wsum, len(complete))
