import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox

# =====================================================
# PERFORMANCE SETTINGS
# =====================================================
tf.keras.backend.set_floatx('float32')

# =====================================================
# FILE PATHS (UPDATED)
# =====================================================
base_path = r"C:\Pradeep\CU\paper\New folder\ANN Check"

model_path_c     = base_path + r"\ann_classification.keras"
model_path_alpha = base_path + r"\ann_alpha.keras"
model_path_kae   = base_path + r"\ann_kae.keras"
file_path        = base_path + r"\data_full.xlsx"

# =====================================================
# LOAD MODELS
# =====================================================
class_model = load_model(model_path_c)
alpha_model = load_model(model_path_alpha)
kae_model   = load_model(model_path_kae)

# Warm-up (faster prediction)
dummy = np.zeros((1, 10), dtype=np.float32)
class_model.predict(dummy)
alpha_model.predict(dummy)
kae_model.predict(dummy)

# =====================================================
# TRAINED INPUT FEATURES
# =====================================================
trained_cols = [
    'Phi', 'Theta', 'Beta', 'Delta/Phi', 'c/(γH)', 'cw/c',
    'q/(γH)', 'P/(γH²)', 'kh', 'kv'
]

data_class = pd.read_excel(file_path, sheet_name="C", usecols=trained_cols)
data_reg   = pd.read_excel(file_path, sheet_name="R", usecols=trained_cols)

scaler_class = StandardScaler().fit(data_class)
scaler_alpha = StandardScaler().fit(data_reg)
scaler_kae   = StandardScaler().fit(data_reg)

print("ANN models and scalers loaded successfully.")

# =====================================================
# SAFE DIVISION (ZERO ACCEPTED)
# =====================================================
def safe_div(a, b):
    if a == 0 or b == 0:
        return 0.0
    return a / b

# =====================================================
# GUI SETUP
# =====================================================
root = tk.Tk()
root.title("Artificial Neural Network (ANN)")
root.geometry("520x800")
root.resizable(False, False)

title = tk.Label(
    root,
    text="Artificial Neural Network (ANN)\nClassification + Alpha + Kae",
    font=("Arial", 15, "bold")
)
title.pack(pady=12)

frame = tk.Frame(root)
frame.pack(pady=10)

entries = {}

raw_cols = [
    'H', 'Phi', 'Theta', 'Beta', 'Delta', 'γ',
    'c', 'cw', 'q', 'P', 'kh', 'kv'
]

for col in raw_cols:
    row = tk.Frame(frame)
    row.pack(pady=4)
    label = tk.Label(row, text=col, width=15, anchor="w", font=("Arial", 11))
    label.pack(side=tk.LEFT)
    ent = tk.Entry(row, width=20, font=("Arial", 11))
    ent.pack(side=tk.RIGHT)
    entries[col] = ent

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_ann():
    try:
        vals = {col: float(entries[col].get()) for col in raw_cols}
    except:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")
        return

    H     = vals['H']
    Phi   = vals['Phi']
    Theta = vals['Theta']
    Beta  = vals['Beta']
    Delta = vals['Delta']
    gamma = vals['γ']
    c     = vals['c']
    cw    = vals['cw']
    q     = vals['q']
    P     = vals['P']
    kh    = vals['kh']
    kv    = vals['kv']

    # Raw → ANN model inputs
    X = np.array([[ 
        Phi,
        Theta,
        Beta,
        safe_div(Delta, Phi),
        safe_div(c, gamma * H),
        safe_div(cw, c),
        safe_div(q, gamma * H),
        safe_div(P, gamma * H * H),
        kh,
        kv
    ]], dtype=np.float32)

    # Scaling
    Xc = scaler_class.transform(X)
    Xa = scaler_alpha.transform(X)
    Xk = scaler_kae.transform(X)

    # Predictions
    cls_prob = float(class_model.predict(Xc, verbose=0)[0][0])
    cls_label = "Feasible (1)" if cls_prob > 0.5 else "Not Feasible (0)"

    alpha_pred = float(alpha_model.predict(Xa, verbose=0)[0][0])
    kae_pred   = float(kae_model.predict(Xk, verbose=0)[0][0]) / 100.00

    # Output
    output = (
        "===== ANN CLASSIFICATION =====\n"
        f"Probability : {cls_prob:.5f}\n"
        f"Label       : {cls_label}\n\n"
        "===== ANN REGRESSION =====\n"
        f"Alpha (α)   : {alpha_pred:.4f}\n"
        f"Kae         : {kae_pred:.4f}\n"
    )

    result_box.config(state="normal")
    result_box.delete(1.0, tk.END)
    result_box.insert(tk.END, output)
    result_box.config(state="disabled")

# =====================================================
# BUTTON & OUTPUT BOX
# =====================================================
predict_btn = tk.Button(
    root,
    text="ANN PREDICT",
    command=predict_ann,
    bg="#0066CC",
    fg="white",
    font=("Arial", 12, "bold"),
    width=22,
    height=2
)
predict_btn.pack(pady=18)

result_label = tk.Label(root, text="Output:", font=("Arial", 12, "bold"))
result_label.pack()

result_box = tk.Text(root, height=12, width=60, font=("Arial", 11))
result_box.pack(pady=10)
result_box.config(state="disabled")

root.mainloop()
