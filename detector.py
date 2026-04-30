"""
╔══════════════════════════════════════════════════════════════════╗
║          Network Intrusion Detection System (NIDS)              ║
║          ML-powered anomalous traffic classifier                 ║
║          Author : Your Name                                      ║
║          Stack  : Python · Scikit-learn · Pandas · NumPy        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
import time
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ─────────────────────────────────────────────────────────────────
#  ANSI colour helpers
# ─────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner():
    print(f"""{CYAN}{BOLD}
  ███╗   ██╗██╗██████╗ ███████╗
  ████╗  ██║██║██╔══██╗██╔════╝
  ██╔██╗ ██║██║██║  ██║███████╗
  ██║╚██╗██║██║██║  ██║╚════██║
  ██║ ╚████║██║██████╔╝███████║
  ╚═╝  ╚═══╝╚═╝╚═════╝ ╚══════╝
  Network Intrusion Detection System v1.0
  Author: Your Name | Cybersecurity Portfolio
{RESET}""")

# ─────────────────────────────────────────────────────────────────
#  FEATURE COLUMNS  (KDD Cup 99 / NSL-KDD schema)
# ─────────────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CAT_COLS    = ["protocol_type", "service", "flag"]
TARGET_COL  = "label"
MODEL_PATH  = "models/nids_model.pkl"
SCALER_PATH = "models/nids_scaler.pkl"
ENC_PATH    = "models/nids_encoders.pkl"

ATTACK_CLASSES = {
    "normal": "NORMAL",
    "dos":    "DOS ATTACK",
    "probe":  "PROBE / SCAN",
    "r2l":    "REMOTE TO LOCAL",
    "u2r":    "USER TO ROOT",
}

# ─────────────────────────────────────────────────────────────────
#  DATA UTILITIES
# ─────────────────────────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic KDD-like traffic for demo purposes.
    Replace with real NSL-KDD dataset for production use.
    """
    np.random.seed(42)
    rng = np.random

    n_normal = int(n_samples * 0.6)
    n_attack = n_samples - n_normal

    def normal_traffic(n):
        d = {c: rng.exponential(1.0, n) for c in NUMERIC_COLS}
        d["src_bytes"] = rng.randint(100, 5000, n).astype(float)
        d["dst_bytes"] = rng.randint(200, 8000, n).astype(float)
        d["duration"]  = rng.randint(0, 60, n).astype(float)
        d["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n, p=[0.7, 0.2, 0.1])
        d["service"]   = rng.choice(["http", "ftp", "ssh", "smtp", "dns"], n)
        d["flag"]      = rng.choice(["SF", "S0", "REJ", "RSTO", "RSTR"], n, p=[0.8,0.05,0.05,0.05,0.05])
        d[TARGET_COL]  = "normal"
        return pd.DataFrame(d)

    def attack_traffic(n):
        d = {c: rng.exponential(5.0, n) for c in NUMERIC_COLS}
        d["src_bytes"] = rng.randint(0, 100, n).astype(float)   # DoS: tiny or zero
        d["dst_bytes"] = rng.randint(0, 50, n).astype(float)
        d["duration"]  = rng.randint(0, 1, n).astype(float)
        d["serror_rate"] = rng.uniform(0.7, 1.0, n)
        d["same_srv_rate"] = rng.uniform(0.9, 1.0, n)
        d["count"] = rng.randint(400, 512, n).astype(float)
        d["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n, p=[0.4, 0.3, 0.3])
        d["service"]   = rng.choice(["http", "ftp_data", "private", "other"], n)
        d["flag"]      = rng.choice(["S0", "REJ", "RSTO", "SF"], n, p=[0.4, 0.3, 0.2, 0.1])
        attack_types   = ["dos", "probe", "r2l", "u2r"]
        d[TARGET_COL]  = rng.choice(attack_types, n, p=[0.6, 0.25, 0.1, 0.05])
        return pd.DataFrame(d)

    df = pd.concat([normal_traffic(n_normal), attack_traffic(n_attack)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def preprocess(df: pd.DataFrame, encoders: dict = None, scaler=None, fit: bool = False):
    """Encode categoricals and scale numeric features."""
    if fit:
        encoders = {}
        for col in CAT_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        scaler = StandardScaler()
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    else:
        for col in CAT_COLS:
            le = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    features = NUMERIC_COLS + CAT_COLS
    X = df[features]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y, encoders, scaler


# ─────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────
def train(data_path: str = None):
    print(f"\n{CYAN}[*] Loading dataset...{RESET}")

    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"{GREEN}[+] Loaded {len(df):,} records from {data_path}{RESET}")
    else:
        print(f"{YELLOW}[!] No dataset path provided — generating synthetic data{RESET}")
        df = generate_synthetic_data(n_samples=8000)
        print(f"{GREEN}[+] Generated {len(df):,} synthetic records{RESET}")

    print(f"\n{CYAN}[*] Preprocessing...{RESET}")
    X, y, encoders, scaler = preprocess(df.copy(), fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print(f"\n{CYAN}[*] Training Random Forest Classifier...{RESET}")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"{GREEN}[+] Training complete in {elapsed:.1f}s{RESET}")

    # Evaluate
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n{BOLD}{'─'*55}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"{'─'*55}{RESET}")
    print(classification_report(y_test, y_pred))

    # Persist
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,    MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(encoders, ENC_PATH)
    print(f"\n{GREEN}[+] Model saved to {MODEL_PATH}{RESET}")

    # Feature importance
    feat_names  = NUMERIC_COLS + CAT_COLS
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:10]
    print(f"\n{CYAN}Top-10 Features:{RESET}")
    for i, idx in enumerate(top_idx, 1):
        bar = "█" * int(importances[idx] * 100)
        print(f"  {i:2}. {feat_names[idx]:<30} {bar} {importances[idx]:.4f}")


# ─────────────────────────────────────────────────────────────────
#  LIVE PREDICTION (simulate streaming packets)
# ─────────────────────────────────────────────────────────────────
def predict_live(n_packets: int = 20):
    if not os.path.exists(MODEL_PATH):
        print(f"{RED}[!] No trained model found. Run: python detector.py train{RESET}")
        sys.exit(1)

    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENC_PATH)

    print(f"\n{CYAN}[*] Starting live packet simulation ({n_packets} packets)...{RESET}\n")
    print(f"  {'Time':<12} {'Protocol':<10} {'Bytes In':>10} {'Bytes Out':>10} {'Result'}")
    print(f"  {'─'*65}")

    alert_log = []

    for i in range(n_packets):
        # Simulate a packet row
        np.random.seed(i * 7)
        is_attack = np.random.random() < 0.35

        row = {c: np.random.exponential(1.0) for c in NUMERIC_COLS}
        row["src_bytes"]  = np.random.randint(0 if is_attack else 100, 200 if is_attack else 6000)
        row["dst_bytes"]  = np.random.randint(0 if is_attack else 200, 100 if is_attack else 8000)
        row["protocol_type"] = np.random.choice(["tcp", "udp", "icmp"])
        row["service"]    = np.random.choice(["http", "ftp", "ssh", "smtp"])
        row["flag"]       = np.random.choice(["S0", "REJ"] if is_attack else ["SF", "SF", "SF"])
        if is_attack:
            row["serror_rate"] = np.random.uniform(0.8, 1.0)
            row["count"]       = np.random.randint(400, 512)
        row[TARGET_COL]   = "?"

        df_row = pd.DataFrame([row])
        X, _, _, _ = preprocess(df_row, encoders=encoders, scaler=scaler, fit=False)
        label = model.predict(X)[0]
        proba = model.predict_proba(X)[0].max()

        ts    = datetime.now().strftime("%H:%M:%S.%f")[:12]
        proto = row["protocol_type"].upper()
        sb    = int(row["src_bytes"])
        db    = int(row["dst_bytes"])

        if label == "normal":
            status = f"{GREEN}✓ NORMAL ({proba:.0%}){RESET}"
        else:
            status = f"{RED}⚠  {ATTACK_CLASSES.get(label, label)} ({proba:.0%}){RESET}"
            alert_log.append({"time": ts, "type": label, "src_bytes": sb})

        print(f"  {ts:<12} {proto:<10} {sb:>10} {db:>10} {status}")
        time.sleep(0.18)

    print(f"\n{YELLOW}{'─'*65}")
    print(f"  Summary: {n_packets} packets | {len(alert_log)} alerts raised")
    if alert_log:
        with open("alerts.json", "w") as f:
            json.dump(alert_log, f, indent=2)
        print(f"  Alerts written to alerts.json")
    print(f"{'─'*65}{RESET}\n")


# ─────────────────────────────────────────────────────────────────
#  CLI ENTRY
# ─────────────────────────────────────────────────────────────────
def main():
    banner()
    parser = argparse.ArgumentParser(description="NIDS — Network Intrusion Detection System")
    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="Train the RF model")
    t.add_argument("--data", default=None, help="Path to CSV dataset (NSL-KDD format)")

    p = sub.add_parser("predict", help="Run live packet simulation")
    p.add_argument("--packets", type=int, default=20, help="Number of packets to simulate")

    args = parser.parse_args()

    if args.cmd == "train":
        train(data_path=args.data)
    elif args.cmd == "predict":
        predict_live(n_packets=args.packets)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
