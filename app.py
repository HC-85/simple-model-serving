from sklearn.linear_model import LinearRegression
from datetime import datetime as dt 
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/suggest_savings', methods=['POST'])
def suggest_savings():
    data = request.get_json(force=True)
    X = np.array(data['X'])
    y = np.array(data['y'])

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)

    dinero_meta = data['dinero_meta']
    fecha_meta = dt.strptime(data['fecha_meta'], '%d-%m-%Y').toordinal()

    dinero_faltante = dinero_meta - y[-1]
    dias_faltantes = fecha_meta - X[-1]

    ahorro_extra_mensual_necesario = (dinero_faltante/dias_faltantes - model.coef_)*30

    return jsonify({'ahorro_extra_mensual_necesario': ahorro_extra_mensual_necesario.item()})


if __name__ == '__main__':
    app.run(debug=True)