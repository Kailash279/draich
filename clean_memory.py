# clean_memory.py
import json

with open("memory.json", "r", encoding="utf-8") as f:
    mem = json.load(f)

cleaned = {k: v for k, v in mem.items() if v and v.strip().lower() not in ["null", "none", ""]}

with open("memory.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print("✅ Cleaned memory.json")
