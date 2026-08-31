with open('backend/static/index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('scratch/emojis_found.txt', 'w', encoding='utf-8') as out:
    for i, line in enumerate(lines, 1):
        has_emoji = any(ord(c) > 8000 for c in line)
        if has_emoji:
            out.write(f"Line {i}: {line.strip()}\n")
print("Done checking emojis.")
