import pickle

import requests
from flask import Flask, render_template, request
from pydantic import BaseModel, Field, PositiveFloat

app = Flask(__name__)
MODEL = pickle.load(open("modelrfr.pkl", "rb"))

@app.route('/', methods=['GET'])

class FormQuery(BaseModel):
    angle_of_incidence: float = Field(..., validation_alias="angle_of_incidence")
    Zenith: float = Field(..., validation_alias="Zenith")
    total_cloud_cover_sfc: float = Field(..., validation_alias="total_cloud_cover_sfc")
    azimuth: float = Field(..., validation_alias="azimuth")
    shortwave_radiation_backwards_sfc: float = Field(..., validation_alias="shortwave_radiation_backwards_sfc")
    relative_humidity_2_m_above_gnd: float = Field(..., validation_alias="relative_humidity_2_m_above_gnd")
    mean_sea_level_pressure_MSL: float = Field(..., validation_alias="mean_sea_level_pressure_MSL")
    wind_gust_10_m_above_gnd: float = Field(..., validation_alias="wind_gust_10_m_above_gnd")
    temperature_2_m_above_gnd: float = Field(..., validation_alias="temperature_2_m_above_gnd")


@app.route("/", methods=["GET"])
def solarenergy_index():
    return render_template("index.html")


@app.route("/predict/", methods=["POST"])
def local_model_result():
    form_query = FormQuery(**request.form.to_dict(flat=True))

    reg = MODEL.predict(
        [
            [
                form_query.angle_of_incidence,
                form_query.Zenith,
                form_query.total_cloud_cover_sfc,
                form_query.azimuth,
                form_query.shortwave_radiation_backwards_sfc,
                form_query.relative_humidity_2_m_above_gnd,
                form_query.mean_sea_level_pressure_MSL,
                form_query.wind_gust_10_m_above_gnd,
                form_query.temperature_2_m_above_gnd,
            ]
        ]
    )[0]
    return render_template("prediction.html", kwvalue=reg)


@app.route("/predict_from_api/", methods=["POST"])
def api_result():
    model_list = requests.get("http://127.0.0.1:8000/model/list/").json()
    if len(model_list) == 0:
        raise Exception("No model could be retrieved from the model registry")

    best_model = sorted(model_list, key=lambda d: d["mse"], reverse=True)[0]
    app.logger.debug(f"Best model retrieved : {best_model}")

    api_response = requests.post(
        "http://127.0.0.1:8000/model/predict/",
        json={
            **{"train_id": best_model["train_id"]},
            **FormQuery(**request.form.to_dict(flat=True)).model_dump(),
        },
    )

    response = api_response.json()
    app.logger.debug(response)

    return render_template("prediction.html", kwvalue=response["reg"])

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000, debug=True)