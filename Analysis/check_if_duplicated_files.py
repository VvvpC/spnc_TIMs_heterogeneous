from pathlib import Path
import re
from collections import Counter, defaultdict

def stat_random_points_folder(
    folder: str,
    expected_tasks=("KRandGR", "MC", "NARMA10"),
    exts=(".csv", ".pkl"),
):
    """
    统计 folder 下 result_{TASK}_heterogeneous_{ID}_*.csv 的数量，并检查每个 ID 是否缺少 TASK。

    返回：
      - task_counts: Counter({task: count})
      - missing_by_id: dict[int, list[str]]   # 每个缺少的测试
      - duplicate_pairs: dict[tuple(task,id), list[str]]  # 同一(task,id)出现多个文件时，列出文件名
      - ignored_files: list[str]  # 不符合命名规则/扩展名的文件
    """
    folder = Path(folder)

    # 解析：result_KRandGR_heterogeneous_0_....csv
    pat = re.compile(r"^result_(?P<task>[^_]+)_heterogeneous_(?P<id>\d+)_")

    expected_tasks = tuple(expected_tasks)
    expected_set = set(expected_tasks)

    task_counts = Counter()
    id_to_tasks = defaultdict(set)
    pair_to_files = defaultdict(list)
    ignored_files = []

    for p in folder.iterdir():
        if not p.is_file():
            continue
        if exts and p.suffix.lower() not in {e.lower() for e in exts}:
            ignored_files.append(p.name)
            continue

        m = pat.match(p.stem)
        if not m:
            ignored_files.append(p.name)
            continue

        task = m.group("task")
        mid = int(m.group("id"))

        task_counts[task] += 1
        id_to_tasks[mid].add(task)
        pair_to_files[(task, mid)].append(p.name)

    # 缺失检查（只按 expected_tasks 来判定缺什么）
    missing_by_id = {}
    for mid, tasks in id_to_tasks.items():
        missing = sorted(expected_set - set(tasks))
        if missing:
            missing_by_id[mid] = missing

    # 重复检查：同一 (task,id) 对应多个文件
    duplicate_pairs = {
        (task, mid): files
        for (task, mid), files in pair_to_files.items()
        if len(files) > 1
    }

    return task_counts, missing_by_id, duplicate_pairs, ignored_files


def print_stat_report(folder: str):
    task_counts, missing_by_id, duplicate_pairs, ignored_files = stat_random_points_folder(folder)

    print("=== Task counts ===")
    for k in ("KRandGR", "MC", "NARMA10"):
        print(f"{k}: {task_counts.get(k, 0)}")
    # 也把其它 task（如果存在）打印出来
    others = sorted([k for k in task_counts.keys() if k not in {"KRandGR", "MC", "NARMA10"}])
    for k in others:
        print(f"{k}: {task_counts[k]}")

    print("\n=== Missing tests by model id ===")
    if not missing_by_id:
        print("All model ids have all expected tests (KRandGR/MC/NARMA10).")
    else:
        for mid in sorted(missing_by_id.keys()):
            print(f"id {mid} missing: {', '.join(missing_by_id[mid])}")

    print("\n=== Duplicates (same task & id has multiple files) ===")
    if not duplicate_pairs:
        print("No duplicates.")
    else:
        for (task, mid) in sorted(duplicate_pairs.keys(), key=lambda x: (x[1], x[0])):
            print(f"({task}, id={mid}) -> {len(duplicate_pairs[(task, mid)])} files")
            for name in duplicate_pairs[(task, mid)]:
                print(f"  - {name}")

    print("\n=== Ignored files (not matching pattern or not .csv) ===")
    print(f"{len(ignored_files)} files ignored.")
    # 如需看具体哪些被忽略，取消下面两行注释
    # for name in ignored_files:
    #     print("  -", name)


if __name__ == "__main__":
    folder = r"C:\Users\Chen\Desktop\Repository\spnc_TIMs_heterogeneous\random_points_TIMs_hetero"
    print_stat_report(folder)