"""按完整过程做时间前推验证并冻结联合分位模型。

流程严格按资料约定：
1. 折叠以 process 整组切分；训练过程的最大 time 必须小于验证过程。
2. 每折用训练窗完整行锁定中心/正 scale，对同一 lambda 网格联合拟合，
   在验证窗完整行上计未惩罚 pinball；lambda 不在折外复用系数。
3. 选择指标为各折逐分位损失总和 / 全部验证行权重和；同分取较大 lambda。
4. 用所选 lambda 在全开发窗重新锁定中心/scale 并重拟合，随后冻结。
5. 外期逐行重算残差与损失；凸包成员判定在原单位进行，域内给正式预测，
   域外只给无保证标识；缺字段行单列、不参与拟合与评分。
外期 y 的任何改动只影响第 5 步评分，不能回到前面任何一步改变系数。
"""
import math

from .domain import summarize
from .geometry import membership
from .preprocess import fit_standardizer, split_rows
from .quantile import fit_joint, pinball, predict


def validate_config(v):
    """资料结构与切分合法性；非法直接报错，不静默修补。"""
    summarize(v)
    d = len(v['covariates'])
    processes = {r['process'] for r in v['rows']}
    dev = list(v['development_processes'])
    ext = list(v['external_processes'])
    if not dev or set(dev) - processes:
        raise ValueError('开发过程缺失或未出现在资料中')
    if set(ext) - processes:
        raise ValueError('外期过程未出现在资料中')
    if set(dev) & set(ext):
        raise ValueError('开发过程与外期过程不得重叠')
    lamb = [float(x) for x in v['lambda_candidates']]
    if not lamb or any(x < 0.0 for x in lamb):
        raise ValueError('lambda 网格必须为非空且全部非负')
    if lamb != sorted(lamb):
        raise ValueError('lambda 网格必须按升序给出（同分取大者依赖该顺序）')
    max_time = {}
    for r in v['rows']:
        max_time[r['process']] = max(max_time.get(r['process'], -math.inf),
                                     int(r['time']))
    seen: set[str] = set()
    for fold in v['folds']:
        tr, va = fold['train'], fold['validation']
        if len(va) != 1:
            raise ValueError('每折必须恰好指定一个验证过程')
        if set(tr) & set(va) or not set(tr).issubset(processes) \
                or not set(va).issubset(processes):
            raise ValueError('折叠过程集合非法')
        # 前推：首折训练集为初始过程；其后训练集必须恰为此前各折的并集，
        # 验证过程必须是未见的新过程。
        if seen:
            if set(tr) != seen:
                raise ValueError('训练集必须恰为此前各折过程的并集（前推）')
        elif not tr:
            raise ValueError('首折训练过程不能为空')
        if seen & set(va):
            raise ValueError('验证过程不得在更早的折中出现')
        if any(max_time[t] >= max_time[va[0]] for t in tr):
            raise ValueError('验证过程必须晚于全部训练过程（time 前推）')
        if not set(tr).issubset(dev) or va[0] not in dev:
            raise ValueError('折叠只能使用开发过程')
        seen.update(tr)
        seen.update(va)
    if seen != set(dev):
        raise ValueError('折叠未恰好覆盖全部开发过程')
    if any(len(vert) != d for vert in v['domain_vertices']):
        raise ValueError('顶点维度与协变量不一致')
    return True


def _rows_of(rows, processes):
    return [r for r in rows if r['process'] in set(processes)]


def select_lambda(grid, scores):
    """按验证均分选 lambda；网格升序，平局即替换，故同分保留最大 lambda。

    grid 与 scores 的键一致且 grid 升序。inf 分（折拟合失败）不参与选择，
    全部失败时抛错而不是静默返回。
    """
    finite = [l for l in grid if scores[l] != math.inf]
    if not finite:
        raise ValueError('全部 lambda 的折拟合都未达最优，无法选择')
    selected = finite[0]
    best = scores[selected]
    for lamb in finite[1:]:
        if scores[lamb] <= best:
            best, selected = scores[lamb], lamb
    return selected


def _fold_fit(v, fold, lamb, standardizer, tol, max_iter):
    d = len(v['covariates'])
    train_complete, _ = split_rows(_rows_of(v['rows'], fold['train']), d)
    z = [standardizer.transform(r['x']) for r in train_complete]
    y = [float(r['y']) for r in train_complete]
    w = [float(r['weight']) for r in train_complete]
    verts_z = [standardizer.transform(vert) for vert in v['domain_vertices']]
    return fit_joint(v['quantiles'], z, y, w, verts_z, lamb,
                     standardizer.constant, tol, max_iter)


def run_pipeline(v):
    validate_config(v)
    d = len(v['covariates'])
    tol = float(v['solver']['tolerance'])
    max_iter = int(v['solver']['max_iterations'])
    taus = [float(q) for q in v['quantiles']]
    grid = [float(x) for x in v['lambda_candidates']]

    # ---- 开发内逐折前推验证 ----
    fold_reports = []
    val_weight_total = 0.0
    fold_loss_sums = {l: 0.0 for l in grid}
    failed_lambda = {l: False for l in grid}
    for fold in v['folds']:
        train_rows = _rows_of(v['rows'], fold['train'])
        val_rows_all = _rows_of(v['rows'], fold['validation'])
        scaler = fit_standardizer(train_rows, d)
        val_complete, val_incomplete = split_rows(val_rows_all, d)
        val_w = [float(r['weight']) for r in val_complete]
        val_weight_total += sum(val_w)
        per_lambda = []
        for lamb in grid:
            fit = _fold_fit(v, fold, lamb, scaler, tol, max_iter)
            if fit.status != 'optimal':
                failed_lambda[lamb] = True
                per_lambda.append({'lambda': lamb, 'status': fit.status,
                                   'validation_pinball_sum': None})
                continue
            loss = 0.0
            for r in val_complete:
                preds = predict(fit, scaler.transform(r['x']))
                for k, tau in enumerate(taus):
                    loss += float(r['weight']) * pinball(
                        tau, float(r['y']) - preds[k])
            fold_loss_sums[lamb] += loss
            per_lambda.append({'lambda': lamb, 'status': 'optimal',
                               'validation_pinball_sum': loss,
                               'lp': fit.solver})
        fold_reports.append({
            'train_processes': list(fold['train']),
            'validation_process': fold['validation'][0],
            'train_complete_rows': scaler.complete_rows,
            'validation_complete_rows': len(val_complete),
            'validation_incomplete_rows': len(val_incomplete),
            'centers': scaler.centers, 'scales': scaler.scales,
            'constant': scaler.constant,
            'lambda_results': per_lambda})

    if val_weight_total <= 0.0:
        raise ValueError('开发验证窗没有完整行，无法选择 lambda')
    cv_score = {l: (math.inf if failed_lambda[l]
                    else fold_loss_sums[l] / val_weight_total)
                for l in grid}
    # 网格升序：select_lambda 平局取较大 lambda；任一折失败的 lambda 排除。
    selected = select_lambda(grid, cv_score)

    # ---- 全开发窗重锁定并重拟合，冻结 ----
    dev_rows = _rows_of(v['rows'], v['development_processes'])
    final_scaler = fit_standardizer(dev_rows, d)
    dev_complete, dev_incomplete = split_rows(dev_rows, d)
    z = [final_scaler.transform(r['x']) for r in dev_complete]
    y = [float(r['y']) for r in dev_complete]
    w = [float(r['weight']) for r in dev_complete]
    verts_z = [final_scaler.transform(vert) for vert in v['domain_vertices']]
    final_fit = fit_joint(taus, z, y, w, verts_z, selected,
                          final_scaler.constant, tol, max_iter)
    if final_fit.status != 'optimal':
        raise RuntimeError(f'最终拟合未达最优：{final_fit.status}')

    # ---- 外期评分：域内正式、域外无保证、缺字段单列 ----
    ext_rows = _rows_of(v['rows'], v['external_processes'])
    ext_complete, ext_incomplete = split_rows(ext_rows, d)
    row_reports, inside_reports, outside_reports = [], [], []
    inside_w = 0.0
    for r in ext_complete:
        member = membership(r['x'], v['domain_vertices'], tol, max_iter)
        preds = predict(final_fit, final_scaler.transform(r['x']))
        entry = {
            'id': r['id'], 'process': r['process'], 'time': r['time'],
            'x_raw': list(r['x']), 'y': float(r['y']),
            'weight': float(r['weight']),
            'in_domain': member.inside,
            'domain_witness': member.witness,
            'domain_residual': None if math.isnan(member.equality_residual)
            else member.equality_residual,
            'domain_lp_status': member.status,
            'predictions': {str(taus[k]): preds[k] for k in range(len(taus))},
            'guarantee': 'in_domain_noncrossing' if member.inside
            else 'outside_domain_no_guarantee'}
        losses, residuals = {}, {}
        for k, tau in enumerate(taus):
            rr = float(r['y']) - preds[k]
            residuals[str(tau)] = rr
            losses[str(tau)] = float(r['weight']) * pinball(tau, rr)
        entry['residuals'] = residuals
        entry['pinball_losses'] = losses
        row_reports.append(entry)
        (inside_reports if member.inside else outside_reports).append(entry)
        if member.inside:
            inside_w += float(r['weight'])

    def summarize_score(entries):
        out = {}
        wsum = sum(e['weight'] for e in entries)
        for k, tau in enumerate(taus):
            key = str(tau)
            loss = sum(e['pinball_losses'][key] for e in entries)
            cover_num = sum(e['weight'] for e in entries
                            if e['y'] <= e['predictions'][key] + 1e-12)
            out[key] = {
                'pinball_sum': loss,
                'pinball_per_weight': loss / wsum if wsum > 0 else None,
                'weighted_coverage': (cover_num / wsum) if wsum > 0 else None,
                'effective_weight': wsum,
                'effective_rows': len(entries)}
        return out

    return {
        'quantiles': taus,
        'cv': {'fold_reports': fold_reports,
               'validation_weight_total': val_weight_total,
               'score_per_lambda': cv_score,
               'selected_lambda': selected,
               'tie_rule': '同分选择较大 lambda；网格按升序'},
        'final_model': {
            'lambda': selected,
            'centers': final_scaler.centers,
            'scales': final_scaler.scales,
            'constant_covariates': final_scaler.constant,
            'vertices_standardized': verts_z,
            'coefficients': {str(taus[k]): final_fit.coefficients[k]
                             for k in range(len(taus))},
            'coefficient_note': 'LP 解可能不唯一；这是声明误差内目标与约束'
                                '均有效的一组合法系数，不声称唯一。',
            'objective': final_fit.objective,
            'pinball_total': final_fit.pinball_total,
            'penalty': final_fit.penalty,
            'solver': final_fit.solver,
            'vertex_gaps': final_fit.vertex_gaps,
            'min_vertex_gap': final_fit.min_vertex_gap,
            'rank': final_fit.rank,
            'rank_deficient': final_fit.rank_deficient,
            'diagnostics': {
                'constant_covariates': [
                    v['covariates'][j] for j, c in enumerate(final_scaler.constant)
                    if c],
                'rank_deficient': final_fit.rank_deficient,
                'design_rank': final_fit.rank,
                'design_columns': len(v['covariates']) + 1,
                'design_rows': len(dev_complete),
                'development_incomplete_rows': len(dev_incomplete),
                'development_incomplete_row_ids':
                    [r['id'] for r in dev_incomplete]}},
        'external': {
            'complete_rows': len(ext_complete),
            'incomplete_rows': len(ext_incomplete),
            'incomplete_row_ids': [r['id'] for r in ext_incomplete],
            'in_domain_rows': len(inside_reports),
            'outside_domain_rows': len(outside_reports),
            'in_domain': summarize_score(inside_reports),
            'outside_domain_auxiliary': summarize_score(outside_reports),
            'coverage_note': '加权覆盖是聚合经验量，不是逐样本区间保证；'
                             'lambda 选择不确定性未计入。',
            'rows': row_reports},
    }
