from pathlib import Path
import re
from urllib.parse import unquote, urlparse


repo_root = Path(__file__).resolve().parent.parent
link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
missing = []

for readme_name in ("README.md", "README.zh-CN.md"):
    readme_path = repo_root / readme_name
    if not readme_path.is_file():
        missing.append(f"{readme_name}: missing README")
        continue

    for match in link_pattern.finditer(readme_path.read_text(encoding="utf-8")):
        target = match.group(1).strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        target = unquote(target.split("#", maxsplit=1)[0])
        parsed = urlparse(target)
        if (
            not target
            or target.startswith("#")
            or parsed.scheme in {"http", "https"}
            or target.startswith("data:image/")
        ):
            continue

        local_path = (repo_root / target).resolve()
        try:
            local_path.relative_to(repo_root)
        except ValueError:
            missing.append(f"{readme_name}: outside repository {target}")
            continue
        if not local_path.exists():
            missing.append(f"{readme_name}: missing {target}")

if missing:
    raise SystemExit("README link check failed:\n" + "\n".join(missing))
