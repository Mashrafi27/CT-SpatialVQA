import json
import re
import sys
from pathlib import Path


def fix_item(item):
    if item.get("is_correct") is None:
        reasoning = item.get("reasoning", "")
        m = re.search(r'"is_correct"\s*:\s*(true|false)', reasoning)
        if m:
            item["is_correct"] = (m.group(1) == "true")
    return item


def main():
    if len(sys.argv) != 3:
        print("usage: fix_gemini_parse_errors.py INPUT OUTPUT")
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    text = inp.read_text()

    is_jsonl = text.lstrip().startswith("{")

    if is_jsonl:
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(fix_item(item))
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        data = json.loads(text)
        data = [fix_item(x) for x in data]
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    print(f"done: {out}")


if __name__ == "__main__":
    main()
