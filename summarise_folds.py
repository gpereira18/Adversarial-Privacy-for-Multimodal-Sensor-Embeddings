import argparse
import re
import statistics
from pathlib import Path

EPOCH_RE = re.compile(
    r"task all=([\d.]+) \(imu=([\d.]+) rgb=([\d.]+)\).*?"
    r"probe all=([\d.]+) \(imu=([\d.]+) rgb=([\d.]+)\)"
)
ATTACK_RE = re.compile(r"(?:final=[\d.]+\s+)?best=([\d.]+)(?:\s+\(epoch \d+\))?\s+majority-class=([\d.]+)")
ATTACK_MOD_RE = re.compile(r"^\s+(imu|rgb): ([\d.]+)$", re.M)


def parse_train(path, last_n=12):
    rows = [m.groups() for m in EPOCH_RE.finditer(Path(path).read_text())]
    if not rows:
        return None
    tail = [[float(v) for v in r] for r in rows[-last_n:]]
    cols = list(zip(*tail))
    return {
        "task_all": statistics.mean(cols[0]),
        "task_imu": statistics.mean(cols[1]),
        "task_rgb": statistics.mean(cols[2]),
        "epochs": len(rows),
    }


def parse_attack(path):
    text = Path(path).read_text()
    m = ATTACK_RE.search(text)
    if not m:
        return None
    out = {"attack_all": float(m.group(1)), "chance": float(m.group(2))}
    for mod, val in ATTACK_MOD_RE.findall(text):
        out[f"attack_{mod}"] = float(val)
    return out


def summarise(label, train_glob, attack_glob):
    rows = []
    for tp in sorted(Path(".").glob(train_glob)):
        fold = re.search(r"fold(\d+)", tp.name)
        fold = fold.group(1) if fold else "?"
        t = parse_train(tp)
        ap = sorted(Path(".").glob(attack_glob.replace("*", fold)))
        a = parse_attack(ap[0]) if ap else None
        if t:
            rows.append((fold, t, a))

    if not rows:
        print(f"{label}: no logs matched {train_glob}")
        return

    print(f"\n=== {label} ({len(rows)} folds) ===")
    print(f"{'fold':<6}{'task_all':>10}{'task_imu':>10}{'task_rgb':>10}"
          f"{'atk_all':>10}{'atk_imu':>10}{'atk_rgb':>10}")
    for fold, t, a in rows:
        a = a or {}
        print(f"{fold:<6}{t['task_all']:>10.3f}{t['task_imu']:>10.3f}{t['task_rgb']:>10.3f}"
              f"{a.get('attack_all', float('nan')):>10.3f}"
              f"{a.get('attack_imu', float('nan')):>10.3f}"
              f"{a.get('attack_rgb', float('nan')):>10.3f}")

    def stat(key, source):
        vals = [(t if source == "t" else (a or {})).get(key) for _, t, a in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            return "n/a"
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{statistics.mean(vals):.3f} +/- {sd:.3f}"

    print("-" * 66)
    print(f"task_all   {stat('task_all', 't')}")
    print(f"task_imu   {stat('task_imu', 't')}")
    print(f"task_rgb   {stat('task_rgb', 't')}")
    print(f"attack_all {stat('attack_all', 'a')}")
    print(f"attack_imu {stat('attack_imu', 'a')}")
    print(f"attack_rgb {stat('attack_rgb', 'a')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="lam025")
    p.add_argument("--logdir", default="logs")
    p.add_argument("--attack_prefix", default="attack",
                    help="e.g. 'attack' for logs/attack_fold*.log, "
                         "'llm_attack' for logs/llm_attack_fold*.log")
    args = p.parse_args()
    summarise(
        f"{args.tag} [{args.attack_prefix}]",
        f"{args.logdir}/fold*_{args.tag}.log",
        f"{args.logdir}/{args.attack_prefix}_fold*_{args.tag}.log",
    )
