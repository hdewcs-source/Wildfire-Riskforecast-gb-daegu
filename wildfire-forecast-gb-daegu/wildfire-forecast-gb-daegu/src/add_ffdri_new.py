"""Add the FFDRI_new column to lstm_dataset.csv.

The training sections of total_code.py expect a column named FFDRI_new, which is not part
of data/lstm_dataset.csv. It is a linear function of five columns that are in the file:

FFDRI_new = 5.3012 + 6.6232 * DWI + 0.7657 * FMI + 1.0994 * TMI
            + 0.0906 * sunlight_era5 - 10.6796 * NDVI

Usage
-----
python src/add_ffdri_new.py data/lstm_dataset.csv data/lstm_dataset_with_ffdri_new.csv
"""

import sys

import pandas as pd

COEF = {"DWI": 6.6232, "FMI": 0.7657, "TMI": 1.0994, "sunlight_era5": 0.0906, "NDVI": -10.6796}
INTERCEPT = 5.3012


def main(src, dst):
    df = pd.read_csv(src)
    df["FFDRI_new"] = INTERCEPT + sum(c * df[k] for k, c in COEF.items())
    df.to_csv(dst, index=False)
    print(f"wrote {dst} with {len(df)} rows")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
