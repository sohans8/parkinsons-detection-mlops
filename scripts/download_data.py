
import pandas as pd
import urllib.request
import zipfile
import subprocess
import shutil
import tempfile

from pathlib import Path


URL = "https://archive.ics.uci.edu/static/public/470/parkinson+s+disease+classification.zip"


def main():

    print("Downloading official UCI dataset...")

    output_dir = Path("dataset/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp:

        temp_dir = Path(temp)

        zip_path = temp_dir / "dataset.zip"

        # Download ZIP
        request = urllib.request.Request(
            URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(request, timeout=180) as response:
            zip_path.write_bytes(response.read())

        print("ZIP downloaded successfully!")

        # Extract ZIP
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(temp_dir)

        print("ZIP extracted successfully!")

        # Find RAR file
        rar_files = list(temp_dir.rglob("*.rar"))

        if not rar_files:
            raise FileNotFoundError("RAR file not found inside ZIP.")

        rar_path = rar_files[0]

        # Find 7-Zip
        seven_zip = shutil.which("7z")

        if seven_zip is None:
            possible_path = Path(
                r"C:\Program Files\7-Zip\7z.exe"
            )

            if possible_path.exists():
                seven_zip = str(possible_path)
            else:
                raise FileNotFoundError(
                    "7-Zip not found. Install 7-Zip first."
                )

        # Extract RAR
        subprocess.run(
            [seven_zip, "x", str(rar_path), f"-o{temp_dir}", "-y"],
            check=True
        )

        print("RAR extracted successfully!")

        # Locate CSV
        csv_files = list(temp_dir.rglob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("CSV file not found.")

        csv_path = csv_files[0]

        # Dataset has a two-line header
        data = pd.read_csv(csv_path, header=1)

        # Save original extracted data
        output_file = output_dir / "parkinsons_uci_470.csv"

        data.to_csv(output_file, index=False)

        print("\nDATASET INFORMATION")
        print("-------------------")

        print("Dataset shape:", data.shape)

        print("\nFirst 10 columns:")
        print(data.columns.tolist()[:10])

        print("\nTarget column:")
        print("class" if "class" in data.columns else "Check column names")

        if "class" in data.columns:
            print("\nClass distribution:")
            print(data["class"].value_counts())

        print("\nSaved to:", output_file)
        print("\nDataset download completed!")


if __name__ == "__main__":
    main()
