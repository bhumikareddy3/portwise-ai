"""Central configuration for PortWise AI. Every tunable lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "port_data.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

# Reproducibility
SEED = 42

# Synthetic data size: 5 ports x 90 snapshots = 450 records
N_OBS_PER_PORT = 90
SNAPSHOT_STEP_DAYS = 4          # one snapshot every 4 days -> ~1 year of data
START_DATE = "2025-01-01"

# Problem definition
CONGESTION_THRESHOLD_HOURS = 12.0   # congested  <=>  actual_delay_hours > threshold
HISTORY_WINDOW = 3                  # rolling window (past snapshots) for historical_delay_hours
PRIOR_DELAY_HOURS = 8.0             # value used for a port's very first snapshot (no history yet)

# Noise / data-quality simulation
MISSING_RATE_WEATHER = 0.03
MISSING_RATE_QUEUE = 0.02

# Modelling
TEST_FRACTION = 0.20            # last 20% of time is the hold-out test set
DECISION_THRESHOLD = 0.50       # probability cut-off for the Precision/Recall/F1/confusion matrix
RISK_LOW = 0.35                 # p < 0.35  -> Low
RISK_HIGH = 0.65                # p >= 0.65 -> High, otherwise Medium
