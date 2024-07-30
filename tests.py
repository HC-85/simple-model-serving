import requests

url = 'http://127.0.0.1:5000/predict'

def test_predict():
    data = {'input': [1]}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    assert response.status_code == 200
    assert response.json() == {'prediction': [1.0]}

    data = {'input': [6]}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    assert response.status_code == 200
    assert response.json() == {'prediction': [6.0]}

    data = {'input': [20]}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    assert response.status_code == 200
    assert response.json() == {'prediction': [20.0]}
    