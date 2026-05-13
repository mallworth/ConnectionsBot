import csv
import json
import os
import sys
from collections import OrderedDict
from typing import Optional


def games_csv_to_json(input_path: str, output_path: Optional[str] = None):
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".json"

    games = OrderedDict()

    with open(input_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                game_id = int(row["Game ID"])
            except Exception:
                continue

            games.setdefault(game_id, []).append(row)

    out_games = []

    for game_id in sorted(games.keys()):
        rows = games[game_id]
        grid = [["" for _ in range(4)] for _ in range(4)]
        categories_map = OrderedDict()

        for r in rows:
            word = r.get("Word", "")
            group = r.get("Group Name", "")
            level = int(r.get("Group Level", ""))
            row = int(r.get("Starting Row", 1)) - 1
            col = int(r.get("Starting Column", 1)) - 1

            if 0 <= row < 4 and 0 <= col < 4:
                grid[row][col] = word

            if group not in categories_map:
                categories_map[group] = {"level": level, "words": []}

            categories_map[group]["words"].append(word)

        categories = []
        for name, info in categories_map.items():
            categories.append({"name": name, "level": info["level"], "words": info["words"]})

        out_games.append({"game_id": game_id, "categories": categories, "grid": grid})

    with open(output_path, "w", encoding="utf-8") as outf:
        json.dump(out_games, outf, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) > 2:
        games_csv_to_json(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        games_csv_to_json(sys.argv[1])
    else:
        print("need at least 1 argument")

