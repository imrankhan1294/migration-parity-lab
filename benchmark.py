"""Generate labeled synthetic migration snapshots; no database or third-party packages."""
import argparse
import hashlib
import json
import random

KINDS = ("clean", "reorder", "approved_padding", "rounded_decimal",
         "null_to_empty", "missing_grant", "type_drift", "extra_grant",
         "timezone_shift", "missing_row")


def fixture(seed: int, kind: str) -> dict:
    rng = random.Random(seed)
    rows = [
        {"id": index, "balance": f"{rng.randrange(10, 900)}.{rng.randrange(1, 1000):03}",
         "account": f"ACCT-{rng.randrange(1000, 9999)}", "comment": None if index % 4 == 0 else "demo",
         "opened_at": f"2025-02-10T{index % 24:02}:00:00Z"}
        for index in range(rng.randrange(12, 33))
    ]
    schema = {"id": "bigint", "balance": "numeric(12,3)", "account": "text",
              "comment": "text", "opened_at": "timestamptz"}
    grants = [["analytics", "SELECT"], ["billing", "SELECT"], ["billing", "UPDATE"]]
    source = {"rows": rows, "schema": schema, "grants": grants}
    target = json.loads(json.dumps(source))
    i = rng.randrange(len(rows))
    if kind == "reorder":
        target["rows"].reverse()
        target["grants"].reverse()
    elif kind == "approved_padding":
        target["rows"][i]["account"] += "  "
    elif kind == "rounded_decimal":
        original = target["rows"][i]["balance"]
        target["rows"][i]["balance"] = original[:-1] + ("0" if original[-1] != "0" else "1")
    elif kind == "null_to_empty":
        target["rows"][i // 4 * 4]["comment"] = ""
    elif kind == "missing_grant":
        target["grants"].remove(["billing", "UPDATE"])
    elif kind == "extra_grant":
        target["grants"].append(["analytics", "UPDATE"])
    elif kind == "type_drift":
        target["schema"]["balance"] = "double precision"
    elif kind == "timezone_shift":
        target["rows"][i]["opened_at"] = rows[i]["opened_at"][:-1] + "+01:00"
    elif kind == "missing_row":
        target["rows"].pop(i)
    elif kind != "clean":
        raise ValueError(f"unknown kind {kind!r}")
    bucket = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:2], "big") % 10
    split = "train" if bucket < 6 else "calibration" if bucket < 8 else "test"
    return {"seed": seed, "case": kind, "split": split, "source": source, "target": target,
            "contract": {"key": ["id"], "strip_trailing_ascii_spaces": ["account"]},
            "defect": kind not in ("clean", "reorder", "approved_padding")}


def generate(seed_count: int):
    for seed in range(seed_count):
        for kind in KINDS:
            yield fixture(seed, kind)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    for sample in generate(args.seeds):
        print(json.dumps(sample, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
