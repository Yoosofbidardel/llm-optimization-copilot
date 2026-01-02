import json
from pathlib import Path

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"


def build_example(idx: int) -> dict:
    prompt = (
        f"Assign two tasks t{idx}a and t{idx}b to resources r1 (cap 1) and r2 (cap 2). "
        f"Costs: t{idx}a r1=3, r2=5; t{idx}b r1=4, r2=2. All tasks must be assigned."
    )
    spec = {
        "tasks": [
            {"id": f"t{idx}a", "cost_per_resource": {"r1": 3, "r2": 5}},
            {"id": f"t{idx}b", "cost_per_resource": {"r1": 4, "r2": 2}},
        ],
        "resources": [{"id": "r1", "capacity": 1}, {"id": "r2", "capacity": 2}],
        "must_assign_all_tasks": True,
    }
    return {"prompt": prompt, "json_spec": spec}


def write_dataset(path: Path, examples: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} rows to {path}")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    train_examples = [build_example(i) for i in range(50)]
    val_examples = [build_example(100 + i) for i in range(10)]
    write_dataset(TRAIN_PATH, train_examples)
    write_dataset(VAL_PATH, val_examples)


if __name__ == "__main__":
    main()
