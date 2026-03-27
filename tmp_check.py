import csv

for file_name in ["Faizee_granular_data.csv", "Chiging_data.csv"]:
    try:
        with open(f"/home/Kdixter/Desktop/final_analysis/data/dataset_3/{file_name}", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            classes = set()
            for row in reader:
                if 'class' in row and row['class']:
                    classes.add(row['class'])
            print(f"[{file_name}] Exact classes:\n{sorted(list(classes))}\n")
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
