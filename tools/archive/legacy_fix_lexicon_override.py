from pathlib import Path
p=Path("build_lexicon.py")
s=p.read_text(encoding="utf-8")
s=s.replace('lex["items"].setdefault(k,v)','lex["items"][k]=v')
s=s.replace('"文件":"ไฟล์"','"文件":"ไฟล์"')
p.write_text(s,encoding="utf-8")
