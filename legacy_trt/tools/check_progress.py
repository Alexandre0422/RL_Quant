"""
check_progress.py — 查询远程穷举搜索进度与耗时估算
从 GLAD_Quant/ 目录运行：python tools/check_progress.py
"""
import sys, itertools
sys.path.insert(0, 'tools')
from ssh_helper import run_in_project, put_text

remote_script = """
import json, os, itertools
from collections import defaultdict

path = 'engines/exhaustive_results.json'
if not os.path.exists(path):
    print('FILE NOT FOUND')
    exit()

d = json.load(open(path))
done = [k for k,v in d.items() if v.get('lat')]
lats = sorted([(k,v['lat'],v['spd']) for k,v in d.items() if v.get('lat')], key=lambda x: x[1])

print(f'Done: {len(done)} / 8191')
print('Top 10:')
for k,l,s in lats[:10]:
    print(f'  {s:.4f}x  {l:.4f}ms  {k}')

by_k = defaultdict(list)
for key,v in d.items():
    if v.get('dur'):
        by_k[len(key.split(','))].append(v['dur'])

N = 13
total_remain = 0
rows = []
for k in range(1, N+1):
    total_k = sum(1 for _ in itertools.combinations(range(N), k))
    done_k  = len(by_k[k])
    remain  = total_k - done_k
    avg = sum(by_k[k])/len(by_k[k]) if by_k[k] else (sum(by_k[2])/len(by_k[2])*(1+(k-2)*0.1) if by_k.get(2) else 30)
    est = remain * avg
    total_remain += est
    if remain > 0:
        rows.append(f'  k={k}: {remain} left  ~{est/3600:.1f}h')
if rows:
    print('\\nRemaining:')
    for r in rows:
        print(r)
    print(f'Total: ~{total_remain/3600:.1f}h')
"""

put_text(remote_script, '/tmp/check_prog.py')
out, err, _ = run_in_project('python3 /tmp/check_prog.py', timeout=30)
print(out)
if err.strip():
    print('ERR:', err[:200])
