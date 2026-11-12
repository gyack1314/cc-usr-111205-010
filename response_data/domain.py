"""核对分位集合与原始时延资料的维度。"""
from .io import unique,positive,number

def summarize(v):
    unique(v['rows'],'id');d=len(v['covariates']);missing=0;groups=set()
    qs=v['quantiles']
    if not qs or any(not 0<number(q)<1 for q in qs) or qs!=sorted(set(qs)):raise ValueError('分位水平无效')
    for r in v['rows']:
        if len(r['x'])!=d:raise ValueError('协变量维度无效')
        positive(r['weight']);groups.add(r['process'])
        if r['y'] is None or any(x is None for x in r['x']):missing+=1
        else:
            number(r['y'])
            for x in r['x']:number(x)
    if any(len(x)!=d for x in v['domain_vertices']):raise ValueError('顶点维度无效')
    return {'rows':len(v['rows']),'incomplete_rows':missing,'processes':len(groups),'covariates':d,'quantiles':len(qs)}
