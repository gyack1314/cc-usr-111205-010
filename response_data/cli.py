"""命令行入口：原始摘要、联合拟合流水线与凸包成员见证。"""
import argparse
import json
import math

from .domain import summarize
from .geometry import membership
from .io import load
from .pipeline import run_pipeline


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command',
                   choices=['inspect', 'fit', 'check-domain'])
    p.add_argument('file')
    p.add_argument('--point',
                   help='check-domain 专用：JSON 数组形式的原单位协变量点')
    a = p.parse_args()
    v, digest = load(a.file)

    if a.command == 'inspect':
        _print({'source_sha256': digest, 'summary': summarize(v)})
        return

    if a.command == 'check-domain':
        if a.point is None:
            raise SystemExit('check-domain 需要 --point JSON 数组')
        point = json.loads(a.point)
        if not isinstance(point, list):
            raise SystemExit('--point 必须是数组')
        m = membership(point, v['domain_vertices'],
                       float(v['solver']['tolerance']),
                       int(v['solver']['max_iterations']))
        _print({'source_sha256': digest, 'point': point,
                'inside': m.inside, 'witness': m.witness,
                'equality_residual': None
                if math.isnan(m.equality_residual)
                else m.equality_residual,
                'lp_status': m.status,
                'note': '成员见证仅对给定顶点的凸包有效，不推广到整个实空间'})
        return

    report = run_pipeline(v)
    report = {'source_sha256': digest, **report}
    _print(report)


if __name__ == '__main__':
    main()
