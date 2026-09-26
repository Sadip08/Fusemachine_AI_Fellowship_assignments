from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve


DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)
DATA_PATH = Path(__file__).parents[1] / "data" / "Telco-Customer-Churn.csv"


def main() -> None:
    destination = Path(os.getenv("TELCO_DATA_PATH", DATA_PATH))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Dataset already exists: {destination}")
        return
    urlretrieve(os.getenv("TELCO_DATA_URL", DATA_URL), destination)
    print(f"Downloaded Telco dataset to {destination}")


if __name__ == "__main__":
    main()

