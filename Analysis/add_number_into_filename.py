from pathlib import Path
import re
from collections import defaultdict

# ====== 改成你的目录 ======
dir_k = Path("random_points_TIMs_hetero")   # 例如: Path(r"D:\xxx\KRandGR")
dir_n = Path("random_points_TIMs_hetero")   # 例如: Path(r"D:\xxx\NARMA10")
dry_run = False  # 先 True 预览；确认无误再改 False 真重命名
# =========================

# 严格匹配你给的例子结构
KRANDGR_RE = re.compile(
    r"^result_KRandGR_heterogeneous_(?P<model_id>\d+)_"
    r"n(?P<n>\d+)_sr(?P<sr_from>\d+)to(?P<sr_to>\d+)_"
    r"w(?P<w>[^_]+)_"
    r"tr(?P<tr_from>\d+\.\d+)to(?P<tr_to>\d+\.\d+)s(?P<step>\d+\.\d+)_"
    r"bs(?P<bs>\d+)_bt(?P<bt>[\d.]+)_temp_sweep\.(?P<ext>csv)$"
)

NARMA10_RE = re.compile(
    r"^result_NARMA10_heterogeneous_"
    r"n(?P<n>\d+)_sr(?P<sr_from>\d+)to(?P<sr_to>\d+)_"
    r"w(?P<w>[^_]+)_"
    r"tr(?P<tr_from>\d+\.\d+)to(?P<tr_to>\d+\.\d+)s(?P<step>\d+\.\d+)_"
    r"bs(?P<bs>\d+)_bt(?P<bt>[\d.]+)_temp_sweep\.(?P<ext>pkl)$"
)

# 1) 建立 w -> model_id 映射（检查唯一性）
w_to_ids = defaultdict(set)

k_files = sorted(dir_k.glob("result_KRandGR_heterogeneous_*_temp_sweep.csv"))
for f in k_files:
    m = KRANDGR_RE.match(f.name)
    if not m:
        continue
    w_to_ids[m["w"]].add(m["model_id"])

w_to_id = {}
ambiguous_w = {w: sorted(ids) for w, ids in w_to_ids.items() if len(ids) > 1}
for w, ids in w_to_ids.items():
    if len(ids) == 1:
        w_to_id[w] = next(iter(ids))

print(f"[KRandGR] files matched: {sum(1 for f in k_files if KRANDGR_RE.match(f.name))}/{len(k_files)}")
print(f"[KRandGR] unique w keys: {len(w_to_id)}")
if ambiguous_w:
    print(f"[KRandGR] AMBIGUOUS w keys: {len(ambiguous_w)} (same w -> multiple model_id)")
    for w, ids in list(ambiguous_w.items())[:10]:
        print("  AMBIGUOUS:", w, "->", ids)

# 2) NARMA10 按 w 查 model_id，并重命名：heterogeneous_{model_id}_n...
n_files = sorted(dir_n.glob("result_NARMA10_heterogeneous_*_temp_sweep.pkl"))

renamed, conflicts = 0, 0
unmatched, ambiguous = [], []

for f in n_files:
    m = NARMA10_RE.match(f.name)
    if not m:
        continue

    w = m["w"]

    if w in ambiguous_w:
        ambiguous.append((f.name, w, ambiguous_w[w]))
        continue

    model_id = w_to_id.get(w)
    if model_id is None:
        unmatched.append((f.name, w))
        continue

    # 生成新文件名：把 model_id 插入 heterogeneous_ 后面
    new_name = (
        f"result_NARMA10_heterogeneous_{model_id}_"
        f"n{m['n']}_sr{m['sr_from']}to{m['sr_to']}_w{m['w']}_"
        f"tr{m['tr_from']}to{m['tr_to']}s{m['step']}_bs{m['bs']}_bt{m['bt']}_"
        f"temp_sweep.pkl"
    )

    new_path = f.with_name(new_name)

    if new_path.exists():
        print("CONFLICT (target exists), skip:", new_name)
        conflicts += 1
        continue

    print(f"{f.name}  ->  {new_name}")
    if not dry_run:
        f.rename(new_path)
    renamed += 1

print("\n===== Summary =====")
print("NARMA10 total candidates:", len(n_files))
print("Renamed:", renamed, "(dry_run)" if dry_run else "")
print("Conflicts:", conflicts)
print("Unmatched:", len(unmatched))
print("Ambiguous:", len(ambiguous))

if unmatched:
    print("\n[Unmatched examples]")
    for name, w in unmatched[:10]:
        print("  ", name, "| w =", w)

if ambiguous:
    print("\n[Ambiguous examples]")
    for name, w, ids in ambiguous[:10]:
        print("  ", name, "| w =", w, "| model_ids =", ids)
