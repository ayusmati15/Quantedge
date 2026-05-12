import yfinance as yf
import matplotlib.pyplot as plt
from src.feature_engineering import create_features
from src.ai_model import train_ai_model
def main():

    df = yf.download(
        "^NSEI",
        start="2020-01-01"
    )

    df = create_features(df)
    model, acc, df_test = train_ai_model(df)
    print("\n================ AI MODEL ================\n")
    print(f"Accuracy: {acc:.4f}")
    print("\n==========================================\n")

    plt.figure(figsize=(12, 6))

    plt.plot(
        df_test["AI_Cumulative"],
        label="AI Strategy"
    )
    plt.title("AI Trading Strategy Performance")
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == "__main__":
    main()