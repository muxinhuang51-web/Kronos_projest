from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from model import Kronos, KronosTokenizer, KronosPredictor


def resolve_csv_path() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "XSHG_5min_600977.csv",
        PROJECT_ROOT / "tests" / "data" / "regression_input.csv",
        PROJECT_ROOT / "finetune_csv" / "data" / "HK_ali_09988_kline_5min_all.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "找不到可用的数据文件，请把样例数据放到以下位置之一：\n"
        f"{candidate_text}"
    )


def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax.set_ylabel('Close Price', fontsize=14)
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 3. Prepare Data
csv_path = resolve_csv_path()
print(f"使用数据文件: {csv_path}")
df = pd.read_csv(csv_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
kline_df = df.loc[:lookback+pred_len-1]

# visualize
plot_prediction(kline_df, pred_df)

