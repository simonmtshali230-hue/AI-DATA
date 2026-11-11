import csv
import io
import urllib.request
from datetime import datetime

SEASONS = ["2122", "2223", "2324", "2425", "2526", "2627"]
LEAGUE = "E0"
OUTPUT_FILE = "matches_xg.csv"

all_rows = []

for season in SEASONS:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{LEAGUE}.csv"
    print(f"Downloading {season}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("latin-1")
    except Exception as e:
        print(f"  failed: {e}")
        continue
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        ht = (row.get("HomeTeam") or "").strip()
        at = (row.get("AwayTeam") or "").strip()
        hg = (row.get("FTHG") or "").strip()
        ag = (row.get("FTAG") or "").strip()
        hst = (row.get("HST") or "").strip()
        ast = (row.get("AST") or "").strip()
        date_str = (row.get("Date") or "").strip()
        if not (ht and at and hg and ag and hst and ast and date_str):
            continue
        dt = None
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                dt = datetime.strptime(date_str, fmt); break
            except ValueError:
                continue
        if dt is None:
            continue
        all_rows.append([dt.strftime("%Y-%m-%d"), ht, at,
                         int(float(hg)), int(float(ag)),
                         int(float(hst)), int(float(ast))])

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "home_team", "away_team", "home_goals", "away_goals", "home_sot", "away_sot"])
    w.writerows(all_rows)

print(f"Created {OUTPUT_FILE} with {len(all_rows)} matches.")
print(f"Latest match date in file: {all_rows[-1][0] if all_rows else 'none'}")