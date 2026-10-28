from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).resolve().parent / "data"
FEATURES = ["cicek_say", "temperatur", "rutubet"]
NEW_DAY = pd.DataFrame({"cicek_say": [180], "temperatur": [26], "rutubet": [55]})


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "R²": r2_score(actual, predicted)
    }


def analyze_file(filename):
    df = pd.read_csv(DATA_DIR / filename)
    x = df[FEATURES]
    y = df["bal_kq"]
    results = []

    for model_name in ["Multiple Linear", "Polynomial"]:
        if model_name == "Polynomial":
            transformer = PolynomialFeatures(degree=2, include_bias=False)
            training_data = transformer.fit_transform(x)
            prediction_data = transformer.transform(NEW_DAY)
            feature_names = transformer.get_feature_names_out(FEATURES)
        else:
            training_data = x
            prediction_data = NEW_DAY
            feature_names = FEATURES

        model = LinearRegression().fit(training_data, y)
        predicted = model.predict(training_data)
        future_value = model.predict(prediction_data)[0]
        coefficients = pd.Series(model.coef_, index=feature_names).round(5)

        print(f"\n{filename} - {model_name}")
        print("Əmsallar:")
        print(coefficients.to_string())
        print(f"b: {model.intercept_:.5f}")
        print(f"Yeni gün üçün bal: {future_value:.3f} kq")

        results.append({"Fayl": Path(filename).stem, "Model": model_name,
                        **calculate_metrics(y, predicted)})

    return results


def main():
    rows = []
    for count in (10, 20, 50):
        rows.extend(analyze_file(f"bal_{count}.csv"))

    table = pd.DataFrame(rows)
    print("\nMetriklərin müqayisəsi:")
    print(table.round(4).to_string(index=False))
    table.to_csv(Path(__file__).resolve().parent / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
