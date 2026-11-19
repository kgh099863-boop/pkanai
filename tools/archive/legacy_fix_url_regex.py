from pathlib import Path
p=Path("translate.py")
lines=p.read_text(encoding="utf-8").splitlines()
for i,line in enumerate(lines):
    if line.startswith("URL_OR_EMAIL="):
        lines[i]='URL_OR_EMAIL=re.compile(r"(?:https?://|www[.])[^ \\t\\r\\n]+|[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+[.][A-Za-z]{2,}")'
p.write_text("\n".join(lines)+"\n",encoding="utf-8")
