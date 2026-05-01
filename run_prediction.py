import argparse
from pathlib import Path

import pandas as pd

from model import Kronos, KronosTokenizer, KronosPredictor


def resolve_csv_path(user_csv: str | None) -> Path:
	project_root = Path(__file__).resolve().parent
	default_candidates = [
		project_root / "data" / "XSHG_5min_600977.csv",
		project_root / "finetune_csv" / "data" / "HK_ali_09988_kline_5min_all.csv",
		project_root / "tests" / "data" / "regression_input.csv",
	]

	if user_csv:
		candidate = Path(user_csv).expanduser()
		if not candidate.is_absolute():
			candidate = (project_root / candidate).resolve()
		if candidate.exists():
			return candidate
		raise FileNotFoundError(f"指定的 CSV 不存在: {candidate}")

	for candidate in default_candidates:
		if candidate.exists():
			return candidate

	candidate_lines = "\n".join(f"- {p}" for p in default_candidates)
	raise FileNotFoundError(
		"未找到可用的样例 CSV。请使用 --csv 指定路径，或将数据放到以下任一路径:\n"
		f"{candidate_lines}"
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run Kronos prediction demo")
	parser.add_argument(
		"--csv",
		type=str,
		default=None,
		help="CSV path. Relative path is resolved from project root.",
	)
	args = parser.parse_args()

	# 1. 加载模型（README 指定的 Hugging Face 权重）
	# 3060Ti 跑 small 最稳
	tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
	model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

	# 2. 准备数据（优先用户指定路径，否则自动选择仓库中的可用样例）
	csv_path = resolve_csv_path(args.csv)
	print(f"使用数据文件: {csv_path}")
	df = pd.read_csv(csv_path)
	df["timestamps"] = pd.to_datetime(df["timestamps"])

	# 3. 按照逻辑切分数据
	lookback = 400
	pred_len = 120

	required_cols = {"timestamps", "open", "high", "low", "close"}
	missing_cols = required_cols - set(df.columns)
	if missing_cols:
		missing_text = ", ".join(sorted(missing_cols))
		raise ValueError(f"CSV 缺少必要列: {missing_text}")

	# 兼容无成交量字段的数据
	if "volume" not in df.columns:
		df["volume"] = 0.0
	if "amount" not in df.columns:
		df["amount"] = 0.0

	if len(df) < lookback + pred_len:
		raise ValueError(
			f"数据长度不足: 需要至少 {lookback + pred_len} 行，实际 {len(df)} 行"
		)

	# 提取特征列
	x_df = df.loc[: lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
	x_timestamp = df.loc[: lookback - 1, "timestamps"]
	y_timestamp = df.loc[lookback : lookback + pred_len - 1, "timestamps"]

	# 4. 初始化预测器并运行
	predictor = KronosPredictor(model, tokenizer, max_context=512)
	# 这里会打印进度，让你看到模型正在“写”未来的 K 线
	forecasts = predictor.predict(x_df, x_timestamp, y_timestamp, pred_len=pred_len)

	print("预测成功！前 5 行数据：")
	print(forecasts.head())


if __name__ == "__main__":
	main()