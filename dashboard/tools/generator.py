# dashboard/tools/generator.py
import random, time, os, csv
from datetime import datetime

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "events.csv")
OUT = os.path.abspath(OUT)
CATS = ["Plastic", "Burnable", "Cans", "Bottles", "Others"]

print("Demo generator — appending events to", OUT)
print("Ctrl+C to stop.")
try:
    while True:
        cat = random.choice(CATS)
        conf = round(random.uniform(0.7, 0.98), 2)
        ts = datetime.utcnow().isoformat()
        row = [ts, cat, conf, "simulator", "demo"]
        write_header = not os.path.exists(OUT) or os.stat(OUT).st_size == 0
        with open(OUT, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["ts", "category", "confidence", "source", "note"])
            writer.writerow(row)
        print("Appended:", row)
        time.sleep(random.uniform(0.6, 2.2))
except KeyboardInterrupt:
    print("Stopped.")