from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

car = pd.read_csv("Cleaned Car.csv")
model = pickle.load(open("LinearRegressionMode.pkl", "rb"))


@app.route("/")
def index():
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_type = car["fuel_type"].unique()

    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=year,
        fuel_types=fuel_type,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        company    = request.form.get("company", "").strip()
        car_model  = request.form.get("car_model", "").strip()
        fuel_type  = request.form.get("fuel_type", "").strip()
        year_raw   = request.form.get("year", "").strip()
        kms_raw    = request.form.get("kms_driven", "").strip()

        # Validate required fields
        if not all([company, car_model, fuel_type, year_raw, kms_raw]):
            return "All fields are required.", 400

        year       = int(year_raw)
        kms_driven = int(kms_raw)

        input_df = pd.DataFrame(
            [[car_model, company, year, kms_driven, fuel_type]],
            columns=["name", "company", "year", "kms_driven", "fuel_type"],
        )

        prediction = model.predict(input_df)
        price = max(0, int(prediction[0]))   # clamp negatives to 0
        return str(price)

    except (ValueError, TypeError):
        return "Invalid input. Please check all fields.", 400
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return "Something went wrong. Please try again.", 500


if __name__ == "__main__":
    app.run(debug=True)