import argparse
import csv
import os
import re
from collections import defaultdict


NTU_NAME_RE = re.compile(
    r"^S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<person>\d{3})R(?P<replication>\d{3})A(?P<action>\d{3})$",
    re.IGNORECASE,
)


def _split_ext_list(text: str) -> list[str]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    out: list[str] = []
    for item in items:
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        out.append(item.lower())
    return out


def _first_level_under_root(root: str, path: str) -> str | None:
    rel = os.path.relpath(path, root)
    parts = [p for p in rel.split(os.sep) if p and p not in (".", "..")]
    return parts[0] if parts else None


def _find_action_in_parts(parts: list[str]) -> str | None:
    for part in parts:
        m = re.match(r"^(?:A|a)(\d{1,4})$", part)
        if m:
            return f"A{m.group(1).zfill(3)}"
        m = re.match(r"^(?:action|Action|ACTION)[_-]?(\d{1,4})$", part)
        if m:
            return f"A{m.group(1).zfill(3)}"
        m = re.search(r"(?:^|[_-])A(\d{1,4})(?:$|[_-])", part, flags=re.IGNORECASE)
        if m:
            return f"A{m.group(1).zfill(3)}"
    return None


def _find_person_in_parts(parts: list[str]) -> str | None:
    for part in parts:
        m = re.match(r"^(?:P|p)(\d{1,6})$", part)
        if m:
            return f"P{m.group(1).zfill(3)}"
        m = re.match(r"^(?:person|Person|PERSON)[_-]?(\w+)$", part)
        if m:
            return f"person_{m.group(1)}"
        m = re.match(r"^(?:subject|Subject|SUBJECT)[_-]?(\w+)$", part)
        if m:
            return f"subject_{m.group(1)}"
        m = re.search(r"(?:^|[_-])P(\d{1,6})(?:$|[_-])", part, flags=re.IGNORECASE)
        if m:
            return f"P{m.group(1).zfill(3)}"
    return None


def _infer_from_path(root: str, file_path: str) -> tuple[str | None, str | None]:
    base = os.path.splitext(os.path.basename(file_path))[0]
    m = NTU_NAME_RE.match(base)
    if m:
        return f"A{m.group('action')}", f"P{m.group('person')}"

    rel = os.path.relpath(file_path, root)
    parts = [p for p in rel.split(os.sep) if p and p not in (".", "..")]

    action = _find_action_in_parts(parts)
    person = _find_person_in_parts(parts)

    if action is None or person is None:
        lowered = [p.lower() for p in parts]
        for i, p in enumerate(lowered):
            if p == "skeleton":
                if person is None and i - 1 >= 0:
                    person = parts[i - 1]
                if action is None and i - 2 >= 0:
                    action = parts[i - 2]
                break

    if action is None:
        action = _first_level_under_root(root, file_path)
    if person is None:
        if len(parts) >= 2:
            person = parts[1]

    return action, person


def _iter_candidate_files(root: str, exts: set[str]) -> list[str]:
    candidates: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for name in filenames:
            full = os.path.join(dirpath, name)
            lower = full.lower()
            if "skeleton" in lower:
                candidates.append(full)
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                candidates.append(full)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="数据集根目录（包含 skeleton 文件/文件夹）")
    parser.add_argument(
        "--ext",
        default=".skeleton,.npy,.npz,.json,.pkl,.pickle",
        help="额外要统计的文件扩展名（逗号分隔），例如：.skeleton,.npy",
    )
    parser.add_argument("--out", default="", help="可选：输出 CSV 路径")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    exts = set(_split_ext_list(args.ext))

    files = _iter_candidate_files(root, exts)
    action_to_people: dict[str, set[str]] = defaultdict(set)
    action_to_files: dict[str, int] = defaultdict(int)

    for fp in files:
        action, person = _infer_from_path(root, fp)
        action = action or "UNKNOWN_ACTION"
        person = person or "UNKNOWN_PERSON"
        action_to_people[action].add(person)
        action_to_files[action] += 1

    def action_sort_key(a: str) -> tuple[int, str]:
        m = re.match(r"^A(\d+)$", a)
        return (0, f"{int(m.group(1)):09d}") if m else (1, a)

    rows = []
    for action in sorted(action_to_people.keys(), key=action_sort_key):
        rows.append(
            {
                "action": action,
                "num_people": str(len(action_to_people[action])),
                "num_files": str(action_to_files[action]),
            }
        )

    print(f"root: {root}")
    print(f"matched_files: {len(files)}")
    print("")
    print("action\tpeople\tfiles")
    for r in rows:
        print(f"{r['action']}\t{r['num_people']}\t{r['num_files']}")

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["action", "num_people", "num_files"])
            w.writeheader()
            w.writerows(rows)
        print("")
        print(f"csv_saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
