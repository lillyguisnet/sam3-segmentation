#!/bin/bash
# Regenerate the non-photo fixtures. Needs ffmpeg and Python 3.
set -euo pipefail
cd "$(dirname "$0")"
ffmpeg -loglevel error -y -f lavfi -i "color=c=green:s=320x180:d=2,format=yuv420p" \
  -vf "drawbox=x='20+120*t':y=80:w=16:h=16:color=white:t=fill" -r 15 match.mp4
python3 - <<'PY'
import csv, random
random.seed(7)
with open("customers.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["customer_id", "age", "country", "orders_last_year", "total_spent_eur", "newsletter"])
    for i in range(1, 41):
        w.writerow([f"C{i:04d}", random.randint(19, 74), random.choice(["CA", "FR", "US", "DE"]),
                    random.randint(0, 24), round(random.uniform(15, 2400), 2), random.choice(["yes", "no"])])
PY
cat > meeting.txt <<'TXT'
Ana: Let's start with the budget. We are 8% over on hosting this quarter.
Ben: Mostly the new image pipeline. I can move the batch jobs to night hours.
Ana: Good. Next, hiring. We still need one backend engineer.
Chloe: Two candidates are in final interviews this week.
Ana: Last item, the office move. The new lease starts in March.
Ben: Movers are booked for the last weekend of February.
TXT
mkdir -p scan
for i in 001 002 003; do printf 'placeholder, not a real DICOM file\n' > "scan/slice_$i.dcm"; done
