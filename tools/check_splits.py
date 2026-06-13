import numpy as np

data = np.load("data/processed/human_dataset/splits.npz")
print("X_train[0]:", data["X_train"][0])
print("y_train[0]:", data["y_train"][0])
print("Total treino:", len(data["X_train"]))
print("Total val:", len(data["X_val"]))
print("Total teste:", len(data["X_test"]))

# check if all paths are relative
relative = True
for p in data["X_train"][:20]:
    s = str(p)
    if ":" in s or s.startswith("/"):
        relative = False
        print("ABSOLUTE PATH FOUND:", s)
        break
print("All paths relative:", relative)
