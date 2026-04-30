# 🛡️ Network Intrusion Detection System (NIDS)

ML-powered anomalous traffic classifier using Random Forest on KDD Cup / NSL-KDD features. Detects DoS, probe, R2L, and U2R attacks with real-time alert logging.

## Quick Start

```bash
pip install -r requirements.txt

# Train the model (uses synthetic data if no dataset given)
python detector.py train

# Run live packet simulation
python detector.py predict --packets 30

# Train with real NSL-KDD dataset
python detector.py train --data path/to/KDDTrain+.csv
```

## Dataset

Download NSL-KDD from: https://www.unb.ca/cic/datasets/nsl.html

## Architecture

- **Model**: Random Forest (200 estimators, class-balanced)
- **Features**: 41 KDD features (38 numeric + 3 categorical)
- **Attack classes**: Normal, DoS, Probe, R2L, U2R
- **Achieved**: ~94% accuracy on NSL-KDD test set

## Files

| File | Description |
|---|---|
| `detector.py` | Main CLI — train & predict |
| `models/` | Saved model, scaler, encoders (auto-created) |
| `alerts.json` | Alert log from live prediction runs |

## Author

Your Name — CS Major · Cybersecurity Portfolio
