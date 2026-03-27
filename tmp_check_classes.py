import pandas as pd
for file in ["Chiging_data.csv", "Faizee_granular_data.csv"]:
    path = f"/home/Kdixter/Desktop/final_analysis/data/dataset_3/{file}"
    try:
        df = pd.read_csv(path)
        if "class" in df.columns:
            print(f"[{file}] Exact classes:\n", sorted([str(x) for x in df["class"].unique()]))
        elif "target_class" in df.columns:
            print(f"[{file}] Exact target_classes:\n", sorted([str(x) for x in df["target_class"].unique()]))
        else:
            print(f"[{file}] columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error handling {file}: {e}")
