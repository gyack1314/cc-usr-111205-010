"""端到端与数值性质测试：不交叉保证、可重算性、域边界与一维 LP 独立核对。

所有合成资料在测试内以临时目录写入，脚本本身不依赖 data/ 之外的来源；
读取外部资料的命令统一用普通模块路径（项目本身无第三方依赖）。
"""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from response_data.geometry import membership
from response_data.io import load
from response_data.lp import linprog
from response_data.pipeline import run_pipeline, validate_config
from response_data.preprocess import fit_standardizer, split_rows
from response_data.quantile import design_row, fit_joint, pinball, predict


def _toy_1d():
    """一维、单一开发过程加一个外期过程；凸包为区间 [0,4]。"""
    rows = []
    xs = [0, 1, 2, 3, 4]
    ys = [1.0, 2.0, 2.0, 5.0, 6.0]
    for i, (x, y) in enumerate(zip(xs, ys)):
        rows.append({'id': f'd{i}', 'process': 'pd', 'time': i,
                     'x': [x], 'y': y, 'weight': 1})
    rows.append({'id': 'e0', 'process': 'pe', 'time': 10,
                 'x': [2], 'y': 4.0, 'weight': 1})
    return {
        'covariates': ['x'],
        'rows': rows,
        'quantiles': [0.5, 0.9],
        'domain_vertices': [[0], [4]],
        'development_processes': ['pd'],
        'external_processes': ['pe'],
        'folds': [],
        'lambda_candidates': [0],
        'solver': {'tolerance': 1e-9, 'max_iterations': 20000}}


class OneDimensionalLPTests(unittest.TestCase):
    """用一维小资料手工核对 pinball LP 目标，不经过联合封装。"""

    def test_median_pinball_by_enumeration(self):
        v = _toy_1d()
        rows = [r for r in v['rows'] if r['process'] == 'pd']
        tau = 0.5
        # 仅截距的中位数拟合：枚举候选 b，取最小加权 pinball。
        candidates = sorted(r['y'] for r in rows)
        brute = min(
            sum(pinball(tau, r['y'] - b) for r in rows) for b in candidates)
        # 等价 LP：u-v+b=y，min .5u+.5v，b>=0（此处 y 全正，不影响最优）。
        n = len(rows)
        nvar = 2 * n + 1
        A_eq, b_eq = [], []
        c = [0.0] * nvar
        for i, r in enumerate(rows):
            c[i] = tau
            c[n + i] = 1 - tau
            row = [0.0] * nvar
            row[i] = 1.0
            row[n + i] = -1.0
            row[2 * n] = 1.0
            A_eq.append(row)
            b_eq.append(r['y'])
        res = linprog(c, A_eq=A_eq, b_eq=b_eq)
        self.assertEqual(res.status, 'optimal')
        self.assertAlmostEqual(res.objective, brute, places=9)
        self.assertLessEqual(res.primal_residual, 1e-8)
        self.assertLessEqual(res.duality_gap, 1e-8)

    def test_joint_objective_matches_manual_recompute(self):
        v = _toy_1d()
        d = 1
        sc = fit_standardizer(
            [r for r in v['rows'] if r['process'] == 'pd'], d)
        comp, _ = split_rows(
            [r for r in v['rows'] if r['process'] == 'pd'], d)
        z = [sc.transform(r['x']) for r in comp]
        fit = fit_joint(v['quantiles'], z,
                        [r['y'] for r in comp],
                        [r['weight'] for r in comp],
                        [sc.transform(t) for t in v['domain_vertices']],
                        0, [False], 1e-9, 20000)
        self.assertEqual(fit.status, 'optimal')
        # 用公开系数从零重算目标，必须与报告一致。
        manual = 0.0
        for k, tau in enumerate(fit.taus):
            for i, r in enumerate(comp):
                q = sum(fit.coefficients[k][j] * design_row(z[i])[j]
                        for j in range(2))
                manual += r['weight'] * pinball(tau, r['y'] - q)
        self.assertAlmostEqual(manual, fit.pinball_total, places=9)
        self.assertAlmostEqual(manual, fit.objective, places=9)

    def test_intercept_only_lp_matches_brute_enumeration(self):
        """仅截距联合分位：枚举候选截距（观测值）与 LP 最优目标必须一致，
        且高分位截距不低于低分位，这是与单纯形完全独立的核对。"""
        ys = [1.0, 2.0, 2.0, 5.0, 6.0]
        taus = [0.5, 0.9]
        n = len(ys)

        def col(kind, i):
            base = {('lo', 'u'): 0, ('lo', 'v'): n,
                    ('hi', 'u'): 2 * n + 1,
                    ('hi', 'v'): 3 * n + 2}[(kind[0], kind[1])]
            return base + i

        blo_i, bhi_i = 2 * n, 4 * n + 2
        nvar = 4 * n + 3
        c = [0.0] * nvar
        for i in range(n):
            c[col(('lo', 'u'), i)] = taus[0]
            c[col(('lo', 'v'), i)] = 1 - taus[0]
            c[col(('hi', 'u'), i)] = taus[1]
            c[col(('hi', 'v'), i)] = 1 - taus[1]
        Aeq, beq = [], []
        for i in range(n):
            r = [0.0] * nvar
            r[col(('lo', 'u'), i)] = 1
            r[col(('lo', 'v'), i)] = -1
            r[blo_i] = 1
            Aeq.append(r)
            beq.append(ys[i])
            r = [0.0] * nvar
            r[col(('hi', 'u'), i)] = 1
            r[col(('hi', 'v'), i)] = -1
            r[bhi_i] = 1
            Aeq.append(r)
            beq.append(ys[i])
        Aub = [[0.0] * nvar]
        Aub[0][blo_i] = 1
        Aub[0][bhi_i] = -1
        res = linprog(c, A_ub=Aub, b_ub=[0.0], A_eq=Aeq, b_eq=beq,
                      tolerance=1e-10)
        self.assertEqual(res.status, 'optimal')
        brute = min(
            sum(pinball(taus[0], y - bl) for y in ys)
            + sum(pinball(taus[1], y - bh) for y in ys)
            for bl in set(ys) for bh in set(ys) if bh >= bl)
        self.assertAlmostEqual(res.objective, brute, places=8)
        self.assertGreaterEqual(res.x[bhi_i], res.x[blo_i] - 1e-9)


class NonCrossingTests(unittest.TestCase):
    def setUp(self):
        self.v, _ = load('data/normal.json')
        self.report = run_pipeline(self.v)

    def test_vertex_constraints_hold_at_every_public_vertex(self):
        fm = self.report['final_model']
        coef = {float(k): val for k, val in fm['coefficients'].items()}
        taus = self.report['quantiles']
        for z in fm['vertices_standardized']:
            zfull = design_row(z)
            prev = -math.inf
            for tau in taus:
                q = sum(coef[tau][j] * zfull[j]
                        for j in range(len(zfull)))
                self.assertGreaterEqual(q, prev - 1e-8)
                prev = q

    def test_nonnegativity_extends_into_hull_interior(self):
        # 凸包内部与边界随机点：相邻分位差由顶点非负差的凸组合得到，必非负。
        import random
        fm = self.report['final_model']
        coef = {float(k): val for k, val in fm['coefficients'].items()}
        taus = self.report['quantiles']
        verts = self.v['domain_vertices']
        rng = random.Random(0)
        for _ in range(200):
            lam = [rng.random() for _ in verts]
            s = sum(lam)
            lam = [a / s for a in lam]
            point = [sum(lam[i] * verts[i][j] for i in range(len(verts)))
                     for j in range(len(verts[0]))]
            m = membership(point, verts)
            self.assertTrue(m.inside)
            # 用最终冻结标准化把原单位内点变换后计算各分位预测。
            zfull = design_row([
                (point[j] - fm['centers'][j]) / fm['scales'][j]
                for j in range(len(point))])
            qs = [sum(coef[t][j] * zfull[j] for j in range(len(zfull)))
                  for t in taus]
            for a, b in zip(qs, qs[1:]):
                self.assertGreaterEqual(b - a, -1e-8)

    def test_outside_hull_counterexample_is_flagged_not_guaranteed(self):
        v, _ = load('data/external-outside-domain.json')
        report = run_pipeline(v)
        ext = report['external']
        self.assertEqual(ext['in_domain_rows'], 0)
        self.assertEqual(ext['outside_domain_rows'], 8)
        for row in ext['rows']:
            self.assertFalse(row['in_domain'])
            self.assertEqual(row['guarantee'],
                             'outside_domain_no_guarantee')
            self.assertEqual(row['domain_lp_status'], 'infeasible')

    def test_boundary_vertices_are_members(self):
        for vert in self.v['domain_vertices']:
            m = membership(vert, self.v['domain_vertices'])
            self.assertTrue(m.inside)
            self.assertLessEqual(m.equality_residual, 1e-8)


class PerRowRecomputeTests(unittest.TestCase):
    def setUp(self):
        self.v, _ = load('data/normal.json')
        self.report = run_pipeline(self.v)

    def test_residuals_and_losses_recompute_row_by_row(self):
        fm = self.report['final_model']
        coef = {float(k): val for k, val in fm['coefficients'].items()}
        for row in self.report['external']['rows']:
            z = [(row['x_raw'][j] - fm['centers'][j]) / fm['scales'][j]
                 for j in range(len(row['x_raw']))]
            zfull = design_row(z)
            for tau in self.report['quantiles']:
                q = sum(coef[tau][j] * zfull[j]
                        for j in range(len(zfull)))
                self.assertAlmostEqual(q, row['predictions'][str(tau)],
                                       places=9)
                r = row['y'] - q
                self.assertAlmostEqual(r, row['residuals'][str(tau)],
                                       places=9)
                loss = row['weight'] * pinball(tau, r)
                self.assertAlmostEqual(
                    loss, row['pinball_losses'][str(tau)], places=9)

    def test_coverage_is_aggregate_not_per_sample_claim(self):
        note = self.report['external']['coverage_note']
        self.assertIn('聚合', note)
        for tau, stat in self.report['external']['in_domain'].items():
            rows = [r for r in self.report['external']['rows']
                    if r['in_domain']]
            manual = sum(r['weight'] for r in rows
                         if r['y'] <= r['predictions'][tau] + 1e-12) \
                / sum(r['weight'] for r in rows)
            self.assertAlmostEqual(manual, stat['weighted_coverage'],
                                   places=9)


class IncompleteRowTests(unittest.TestCase):
    def test_missing_field_excluded_from_fit_and_listed(self):
        v, _ = load('data/incomplete-covariate.json')
        report = run_pipeline(v)
        # w0-3 属开发窗：折训练完整行数较 normal 少 1（首折 3 而非 4）。
        first = report['cv']['fold_reports'][0]
        self.assertEqual(first['train_complete_rows'], 3)
        # 外期无缺失；缺失行不出现在外期逐行报告。
        ext_ids = {r['id'] for r in report['external']['rows']}
        self.assertNotIn('w0-3', ext_ids)

    def test_split_helpers_isolate_incomplete(self):
        v, _ = load('data/incomplete-covariate.json')
        comp, inc = split_rows(v['rows'], len(v['covariates']))
        self.assertEqual(len(inc), 1)
        self.assertEqual(inc[0]['id'], 'w0-3')
        self.assertTrue(all(r['id'] != 'w0-3' for r in comp))


class ConstantAndRankTests(unittest.TestCase):
    def test_constant_covariate_diagnosed(self):
        v = _toy_1d()
        for r in v['rows']:
            r['x'] = [7]  # 全部恒定
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'c.json'
            p.write_text(json.dumps(v), encoding='utf8')
            loaded, _ = load(p)
            sc = fit_standardizer(
                [r for r in loaded['rows']
                 if r['process'] == 'pd'], 1)
            self.assertTrue(sc.constant[0])
            self.assertEqual(sc.scales[0], 1.0)
            comp, _ = split_rows(
                [r for r in loaded['rows']
                 if r['process'] == 'pd'], 1)
            z = [sc.transform(r['x']) for r in comp]
            fit = fit_joint(loaded['quantiles'], z,
                            [r['y'] for r in comp],
                            [r['weight'] for r in comp],
                            [sc.transform(t)
                             for t in loaded['domain_vertices']],
                            0, [True], 1e-9, 20000)
            self.assertEqual(fit.status, 'optimal')
            self.assertTrue(fit.constant_covariates[0])
            # 含截距与一列零设计，秩为 1（秩不足）。
            self.assertEqual(fit.rank, 1)
            self.assertTrue(fit.rank_deficient)


class FrozenCoefficientTests(unittest.TestCase):
    def test_external_y_change_cannot_move_coefficients(self):
        v, _ = load('data/normal.json')
        base = run_pipeline(v)
        v2 = copy.deepcopy(v)
        for r in v2['rows']:
            if r['process'] in v2['external_processes']:
                r['y'] = r['y'] + 100.0
        perturbed = run_pipeline(v2)
        self.assertEqual(
            base['final_model']['coefficients'],
            perturbed['final_model']['coefficients'])
        self.assertEqual(
            base['final_model']['centers'],
            perturbed['final_model']['centers'])
        # 但外期评分必须改变。
        self.assertNotEqual(
            base['external']['in_domain'],
            perturbed['external']['in_domain'])


class LambdaSelectionTests(unittest.TestCase):
    def test_tie_prefers_larger_lambda(self):
        from response_data.pipeline import select_lambda
        grid = [0.0, 0.25, 1.0]
        # 前两个严格同分，最大者单独更差：必须在同分二者中取 0.25。
        scores = {0.0: 2.0, 0.25: 2.0, 1.0: 3.0}
        self.assertEqual(select_lambda(grid, scores), 0.25)
        # 三者全同分取最大。
        self.assertEqual(
            select_lambda(grid, {l: 1.5 for l in grid}), 1.0)

    def test_failed_folds_excluded_then_raise_if_all_fail(self):
        from response_data.pipeline import select_lambda
        grid = [0.0, 1.0]
        self.assertEqual(
            select_lambda(grid, {0.0: math.inf, 1.0: 2.0}), 1.0)
        with self.assertRaises(ValueError):
            select_lambda(grid, {0.0: math.inf, 1.0: math.inf})

    def test_pipeline_raises_when_every_fold_fails_to_converge(self):
        v, _ = load('data/normal.json')
        v['solver']['max_iterations'] = 1
        with self.assertRaises(ValueError):
            run_pipeline(v)

    def test_cv_score_formula(self):
        report = run_pipeline(load('data/normal.json')[0])
        total = report['cv']['validation_weight_total']
        for lamb, score in report['cv']['score_per_lambda'].items():
            summed = sum(
                fr['lambda_results'][i]['validation_pinball_sum']
                for fr in report['cv']['fold_reports']
                for i, lr in enumerate(fr['lambda_results'])
                if str(lr['lambda']) == str(float(lamb)))
            self.assertAlmostEqual(score, summed / total, places=9)


class ConfigValidationTests(unittest.TestCase):
    def test_invalid_quantiles_rejected(self):
        v, _ = load('data/invalid-domain.json')
        with self.assertRaises(ValueError):
            validate_config(v)

    def test_negative_lambda_rejected(self):
        v, _ = load('data/normal.json')
        v['lambda_candidates'] = [-1.0]
        with self.assertRaises(ValueError):
            validate_config(v)

    def test_future_validation_process_rejected(self):
        v, _ = load('data/normal.json')
        v['folds'][0]['train'], v['folds'][0]['validation'] = \
            v['folds'][0]['validation'], v['folds'][0]['train']
        with self.assertRaises(ValueError):
            validate_config(v)


class SolverStatusTests(unittest.TestCase):
    def test_iter_limit_not_called_optimal(self):
        # 该例需要多次枢轴；上限给 0 时无法完成，状态不得被报成 optimal。
        res = linprog([-3, -1], A_ub=[[2, 1], [1, 2]], b_ub=[4, 3],
                      max_iterations=0)
        self.assertEqual(res.status, 'iter_limit')
        self.assertIsNone(res.objective)
        self.assertIsNone(res.x)

    def test_unbounded_distinguished_from_optimal(self):
        # min -x1, x1 - x2 <= 1, x1,x2>=0：x1 无界，必须显式报 unbounded。
        res = linprog([-1, 0], A_ub=[[1, -1]], b_ub=[1])
        self.assertEqual(res.status, 'unbounded')
        self.assertIsNone(res.objective)

    def test_infeasible_distinguished(self):
        res = linprog([1], A_ub=[[-1], [1]], b_ub=[-1, 0])
        self.assertEqual(res.status, 'infeasible')


if __name__ == '__main__':
    unittest.main()
