import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import urllib.request
from pathlib import Path

df = pd.read_csv("pdebench_data_urls.csv")
df["PDE"] = df["PDE"].str.lower()

# One Burgers file — contains 10,000 samples internally
# One NS file — contains full simulation data
# Total: ~17GB
TARGETS = [
    ("burgers", "1D_Burgers_Sols_Nu0.001.hdf5"),
    ("ns_incom", "ns_incom_inhom_2d_512-0.h5"),
]

for pde_name, filename in TARGETS:
    row = df[
        (df["PDE"] == pde_name) &
        (df["Filename"] == filename)
    ]
    if row.empty:
        print(f"ERROR: {filename} not found in CSV")
        continue

    row = row.iloc[0]
    dest_dir = Path(f"hpit_benchmark/data/{pde_name}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists():
        print(f"Already exists, skipping: {filename}")
        continue

    print(f"Downloading {filename} (~9-8GB)...")

    def progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            mb = count * block_size / 1e6
            print(f"\r  {pct:.1f}% ({mb:.0f}MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(row["URL"], dest_path, reporthook=progress)
        print(f"\n  Done: {dest_path}")
    except Exception as e:
        print(f"\n  FAILED: {e}")

print("\nAll downloads complete.")
print("Total expected: ~17GB across 2 files.")
print("Next step: run NavierStokes2D dry-run to verify data loads correctly.")