import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

processed_path = Path("data/processed/human_dataset")
print(f"Diretório: {processed_path.resolve()}")
sys.stdout.flush()

image_paths = []
labels = []

for label, cls in [(0, "no_human"), (1, "human")]:
    class_dir = processed_path / cls
    files = list(class_dir.glob("*.png"))
    print(f"{cls}: {len(files)} imagens")
    sys.stdout.flush()
    for f in files:
        image_paths.append(str(f.relative_to(processed_path)))
        labels.append(label)

X = np.array(image_paths)
y = np.array(labels)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

val_ratio = 0.15 / 0.30
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1 - val_ratio, random_state=42, stratify=y_temp
)

np.savez(
    processed_path / "splits.npz",
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
)

print("splits.npz criado!")
print(f"Treino: {len(X_train)} | Val: {len(X_val)} | Teste: {len(X_test)}")
