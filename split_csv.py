import csv
import random
import os
import sys

def split_csv_by_game(input: str, ratio: float):
    base, _ = os.path.splitext(input)
    train_out = base + "_train.csv"
    test_out = base + "_test.csv"

    games = {}
    with open(input, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        for row in reader:
            gid = int(row["Game ID"])
            games.setdefault(gid, []).append(row)

    gids = list(games.keys())
    random.shuffle(gids)
    split_at = int(len(gids) * ratio)
    train_gids = set(gids[:split_at])

    with open(train_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for gid in sorted(gids):
            if gid in train_gids:
                for row in games[gid]:
                    writer.writerow(row)

    with open(test_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for gid in sorted(gids):
            if gid not in train_gids:
                for row in games[gid]:
                    writer.writerow(row)

    return train_out, test_out


if __name__ == "__main__":
    if len(sys.argv) > 2:
        split_csv_by_game(sys.argv[1], float(sys.argv[2]))
    else:
        print("need 2 arguments")
