
import pandas as pd
import os

def convert_csv_to_parquet(input_csv, output_parquet=None):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"CSV file not found: {input_csv}")

    if output_parquet is None:
        output_parquet = input_csv.replace('.csv', '.parquet')

    print(f"Loading CSV from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Converting to Parquet: {output_parquet}")
    df.to_parquet(output_parquet, index=False)
    print(f"✅ Done. Parquet saved to: {output_parquet}")

# Example usage
if __name__ == "__main__":
    convert_csv_to_parquet("GoldCorpus.csv")
