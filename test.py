# Check what the CSV filter actually returns
import pandas as pd
df = pd.read_csv("pdebench_data_urls.csv")
df["PDE"] = df["PDE"].str.lower()
filtered = df[df["PDE"].isin(["burgers", "ns_incom"])]
print(f"Filtered rows: {len(filtered)}")
print(filtered[["PDE","Filename"]].to_string())