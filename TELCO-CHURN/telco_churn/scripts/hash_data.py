from pathlib import Path
import hashlib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
source = PROJECT_ROOT / "data" / "raw" / "telco.csv"


if __name__ == "__main__":
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    print(f"Source: {source}")
    print(f"SHA-256: {source_hash}")
