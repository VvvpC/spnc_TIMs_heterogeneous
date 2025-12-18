from pathlib import Path
import re
import pandas as pd
import numpy as np

def add_number_column_to_pkls(
    folder,
    token="heterogeneous_",
    column_name="number",
    inplace=True,
    out_dir=None,
    on_missing="skip",  # "skip" or "raise"
):
    """
    给 folder 下所有 .pkl 增加一列 column_name，其值取自文件名里 token 后面的数字。
    - 默认直接覆盖原文件（inplace=True）
    - 也可以指定 out_dir 另存
    """
    folder = Path(folder)
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    pat = re.compile(rf"{re.escape(token)}(\d+)")
    n_ok, n_skip = 0, 0

    for pkl_path in folder.rglob("*.pkl"):
        m = pat.search(pkl_path.stem)
        if not m:
            msg = f"[no number] {pkl_path.name}"
            if on_missing == "raise":
                raise ValueError(msg)
            print("skip:", msg)
            n_skip += 1
            continue

        number = int(m.group(1))
        obj = pd.read_pickle(pkl_path)

        if isinstance(obj, pd.DataFrame):
            obj[column_name] = number
        elif isinstance(obj, dict):
            # 如果你的 pkl 存的是 dict，也能加；否则你可以按自己的结构改这里
            obj[column_name] = number
        else:
            raise TypeError(f"Unsupported pkl content type: {type(obj)} in {pkl_path.name}")

        if out_dir:
            save_path = out_dir / pkl_path.name
        else:
            save_path = pkl_path if inplace else pkl_path.with_name(pkl_path.stem + "_with_number.pkl")

        pd.to_pickle(obj, save_path)
        n_ok += 1

    return n_ok, n_skip

if __name__ == "__main__":
    add_number_column_to_pkls(
        folder="random_points_TIMs_hetero",
        token="heterogeneous_",
        column_name="number",
        inplace=True,
        on_missing="skip"
    )