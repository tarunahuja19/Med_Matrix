import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_twin_initialize():
    # 32 values: pathology_vol, edema_vol, healthy_vol, then 29 zeros
    observation = [15.0, 10.0, 1325.0] + [0.0] * 29
    response = client.post("/twin/initialize", json={"observation": observation})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert len(json_data["state_vector"]) == 64
    assert "state_timestamp" in json_data


def test_twin_forecast():
    state_vector = [0.1] * 64
    treatment = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    covariates = [55.0, 1.0, 0.0] + [0.0] * 13
    payload = {
        "state_vector": state_vector,
        "time_horizon": 12.0,
        "n_steps": 12,
        "treatment": treatment,
        "covariates": covariates
    }
    response = client.post("/twin/forecast", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert len(json_data["time_points"]) == 13
    assert len(json_data["volumes_mean"]) == 13
    assert len(json_data["volumes_ci_lower"]) == 13
    assert len(json_data["cognitive_mean"]) == 13


def test_twin_simulate():
    state_vector = [0.1] * 64
    covariates = [55.0, 1.0, 0.0] + [0.0] * 13
    payload = {
        "state_vector": state_vector,
        "covariates": covariates,
        "treatment_names": ["no_treatment", "stupp_protocol"],
        "time_horizon": 12.0
    }
    response = client.post("/twin/simulate", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert "no_treatment" in json_data["comparison"]
    assert "stupp_protocol" in json_data["comparison"]
