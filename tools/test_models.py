import joblib
from pathlib import Path

models_dir = Path("models")
for m in sorted(models_dir.glob("*.pkl")):
    try:
        model = joblib.load(m)
        print(f"{m.name}: loaded OK, type={type(model).__name__}")
    except Exception as e:
        print(f"{m.name}: ERROR - {e}")
