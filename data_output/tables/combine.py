import glob
import os

import pandas as pd

# 1. Get a list of all CSV files in the current folder
csv_files = glob.glob("*.csv")

# 2. Create a new Excel Writer
# 'output_data.xlsx' is the name of the file we will create
with pd.ExcelWriter("output_data.xlsx") as writer:
    for file in csv_files:
        # Read the CSV
        df = pd.read_csv(file)

        # Create a sheet name from the filename (limit to 31 chars for Excel)
        sheet_name = os.path.splitext(file)[0][:31]

        # Write to a specific sheet in the Excel file
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Combined {len(csv_files)} files into output_data.xlsx")
