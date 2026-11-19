from pathlib import Path
for name in ["translate.py","test_suite.py"]:
 p=Path(name); s=p.read_text(encoding="utf-8")
 s=s.replace(r"\\u",r"\u").replace(r"\\U",r"\U").replace(r"\\w",r"\w").replace(r"\\S",r"\S")
 if name=="translate.py":
  s=s.replace('    if text in phrase_map:return phrase_map[text]\n','')
 p.write_text(s,encoding="utf-8")
print("regex escapes fixed")
