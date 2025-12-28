from ucimlrepo import fetch_ucirepo 
import os
  
# fetch dataset 
RAW_DATASET_ID = 29
OUTPUT_DIR = "feature-pipeline"
OUTPUT_FILE = "computer_hardware_clean.parquet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    computer_hardware = fetch_ucirepo(id=29) 

    # Don't Standardize since XGBoost is an ensemble-tree based and split off of thresholds instead of distances.
    df = computer_hardware.data.features 
    df_clean = df.drop(columns=['VendorName', 'ModelName'], errors='ignore')
    df_clean = df_clean.fillna(df_clean.mean()) # Mean Imputation

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df_clean.to_parquet(output_path, index=False)
    print(f"Clean and sent Parquet File to: {output_path}")


if __name__ == "__main__":
    main()

    
    





