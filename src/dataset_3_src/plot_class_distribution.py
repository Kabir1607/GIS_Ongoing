import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_class_distribution():
    input_csv = '/home/Kdixter/Desktop/final_analysis/data/dataset_3/raw_dataset_3.csv'
    output_dir = '/home/Kdixter/Desktop/final_analysis/data/dataset_3/dataset_3_visuals'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        
        if 'class' not in df.columns:
            print("Error: The 'class' column was not found in the dataset.")
            return

        # Calculate class distribution
        class_counts = df['class'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        
        # Plotting
        plt.figure(figsize=(14, 12))
        
        # Use a barplot, sorted by count descending
        sns.barplot(data=class_counts, x='Count', y='Class', palette='viridis')
        
        plt.title('Class Distribution in Dataset 3 (Raw)', fontsize=18, pad=20)
        plt.xlabel('Number of Data Points', fontsize=14)
        plt.ylabel('Class', fontsize=14)
        
        # Add values on the end of the bars
        for index, value in enumerate(class_counts['Count']):
            plt.text(value + 0.5, index, str(value), va='center', fontsize=10)
            
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save output
        output_path = os.path.join(output_dir, 'class_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Successfully generated class distribution plot.")
        print(f"Saved to: {output_path}")

    except Exception as e:
        print(f"Failed to process file: {str(e)}")

if __name__ == '__main__':
    plot_class_distribution()
