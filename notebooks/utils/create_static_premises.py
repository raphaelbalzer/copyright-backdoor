import json
from pathlib import Path
root = Path(__file__).parent.parent.parent

if __name__ == "__main__":
    premises_lines = []
    with open(root / "data" / "2-premises" / "premise-static.txt", "r") as f:
        premise_static = f.read()
    premises = [premise_static.format(i=i) for i in range(0, 242)]
    # load semantic premises an append to premises
    with open(root / "data" / "2-premises" / "semantic" / "premises.jsonl", "r") as f:
        for line in f:
            record = json.loads(line)
            if record["split"] == "eval":
                premises.append(record["premise"])
    for i, premise in enumerate(premises):
        split = "train" if i < 142 else "eval"
        mode = "static" if i < 242 else "semantic"
        premises_lines.append({"i": i, "mode": mode, "premise": premise, "split": split, "word_count": len(premise.split())})

    # building a jsonl file with format {"i", "premise", "split"}
    with open(root / "data" / "2-premises" / "static" / "premises.jsonl", "w") as f_out:
        for line in (premises_lines):
            f_out.write(json.dumps(line) + "\n")
            