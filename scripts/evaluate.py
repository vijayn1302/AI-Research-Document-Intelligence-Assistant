import json
from pathlib import Path
from statistics import mean
from app.graph import ask

DATASET = Path("evaluation/dataset.jsonl")

def normalize(text):
    return " ".join(text.lower().split())

def token_f1(pred, truth):
    p = normalize(pred).split()
    t = normalize(truth).split()
    if not p or not t:
        return 0.0
    common = {}
    for x in p:
        common[x] = common.get(x, 0) + 1
    overlap = 0
    remaining = dict(common)
    for x in t:
        if remaining.get(x, 0):
            overlap += 1
            remaining[x] -= 1
    precision = overlap / len(p)
    recall = overlap / len(t)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def main():
    rows = [json.loads(x) for x in DATASET.read_text(encoding="utf-8").splitlines() if x.strip()]
    results = []
    for row in rows:
        result = ask(row["question"], row.get("top_k", 4))
        prediction = result.get("answer", "")
        results.append({
            "question": row["question"],
            "answer": prediction,
            "ground_truth": row["answer"],
            "f1": token_f1(prediction, row["answer"]),
            "grounded": result.get("grounded", False),
            "grounding_score": result.get("grounding_score", 0),
            "retrieved_sources": [d.metadata.get("source") for d in result.get("documents", [])],
        })

    print("Evaluation")
    print("==========")
    print(f"Samples: {len(results)}")
    print(f"Answer token-F1: {mean(x['f1'] for x in results):.3f}")
    print(f"Grounding rate: {mean(float(x['grounded']) for x in results):.3f}")
    print(f"Mean grounding score: {mean(x['grounding_score'] for x in results):.3f}")

    Path("evaluation/results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
