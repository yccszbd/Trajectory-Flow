import re
import sys
from pathlib import Path

KV_RE = re.compile(r"INFO - ([A-Za-z0-9_\.]+):\s*([^\s]+)")

FIELDS = [
    "dataset",
    "dataname",
    "time_sum",
    "normals_correctness",
    "chamferL1_mean",
    "chamferL1_median",
    "chamferL2_mean",
    "chamferL2_median",
    "F_score_0.01",
    "F_score_0.005",
]


def split_blocks(text):
    chunks = re.split(r"(?=.*INFO - pred_file:)", text)
    for chk in chunks:
        if chk.strip() and "pred_file:" in chk:
            yield chk


def parse_block(block):
    data = {}
    for line in block.splitlines():
        m = KV_RE.search(line)
        if m:
            key, val = m.groups()
            if key in FIELDS:
                data[key] = val
    return data


def parse_log_to_table(log_path, out_path=None):
    log_path = Path(log_path)
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    rows = [parse_block(chk) for chk in split_blocks(text)]
    rows = [r for r in rows if r]

    if not rows:
        print("⚠️ 日志中没有匹配到任何数据")
        return None

    if out_path is None:
        name = log_path.stem
        out_path = log_path.with_name(f"{name}_result.csv")
    else:
        out_path = Path(out_path)

    # 计算每一列的最大宽度(包括表头)
    col_widths = {f: max(len(f), max((len(r.get(f, "")) for r in rows), default=0)) for f in FIELDS}

    with out_path.open("w", encoding="utf-8") as f:
        # 写表头
        header = " | ".join(f.ljust(col_widths[f]) for f in FIELDS)
        f.write(header + "\n")
        f.write("-+-".join("-" * col_widths[f] for f in FIELDS) + "\n")
        # 写数据
        for r in rows:
            line = " | ".join(r.get(f, "").ljust(col_widths[f]) for f in FIELDS)
            f.write(line + "\n")

    print(f"✅ 已写入 {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("training.log")
    name = log_path.stem
    csv_path = log_path.with_name(f"{name}_result.csv")

    parse_log_to_table(log_path, csv_path)
