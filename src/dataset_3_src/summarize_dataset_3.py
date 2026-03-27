import pandas as pd
import os

def summarize_classes():
    input_csv = '/home/Kdixter/Desktop/final_analysis/data/dataset_3/raw_dataset_3.csv'
    output_txt = '/home/Kdixter/Desktop/final_analysis/data/dataset_3/dataset_3_summary.txt'
    
    print(f"Reading {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        
        if 'class' in df.columns and 'class description' in df.columns:
            # Extract unique pairings of class and class description
            unique_data = df[['class', 'class description']].drop_duplicates().dropna(how='all')
            # Sort alphabetically by class for easy reading
            unique_data = unique_data.sort_values(by='class')
            
            lines = []
            lines.append("=========================================================")
            lines.append("Unique Classes and Descriptions in: raw_dataset_3.csv")
            lines.append("=========================================================\n")
            
            for index, row in unique_data.iterrows():
                cls_name = row['class']
                desc = row['class description']
                lines.append(f"Class: {cls_name:<10} | Description: {desc}")
                
            lines.append(f"\nTotal unique classes found: {len(unique_data)}")
            
            os.makedirs(os.path.dirname(output_txt), exist_ok=True)
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")
                
            print(f"Summary successfully written to: {output_txt}")
        else:
            print("Error: The CSV does not contain both 'class' and 'class description' columns. ")
            print(f"Available columns are: {list(df.columns)}")
            
    except Exception as e:
        print(f"Failed to process file: {str(e)}")

if __name__ == '__main__':
    summarize_classes()
