from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Mapeo de tipos de consumidores
consumer_types = {
    1: "Consumidor tradicional",
    2: "Consumidor emocional",
    3: "Consumidor escéptico",
    4: "Consumidor impulsivo",
    5: "Consumidor indeciso"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            temp_directory = os.path.join('static', 'temp')
            if not os.path.exists(temp_directory):
                os.makedirs(temp_directory)

            filepath = os.path.join(temp_directory, file.filename)
            file.save(filepath)
            session['filepath'] = filepath
            data = pd.read_csv(filepath)
            return render_template('train.html', data=data.to_html())
    return render_template('train.html')

@app.route('/train', methods=['GET'])
def train():
    file_path = session.get('filepath')
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
        X = df.iloc[:, 1:]
        y = df.iloc[:, 1]
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.dropna()
        if X.empty or y.empty:
            return "No hay suficientes datos numéricos para entrenar el modelo."

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            model_filename = 'models/consumer_model.pkl'
            if not os.path.exists('models'):
                os.makedirs('models')
            joblib.dump(model, model_filename)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Renderiza una página con la precisión y opciones para volver al inicio o ir a predicción
            return render_template('train_result.html', accuracy=f"Entrenamiento completo. Precisión del modelo: {accuracy * 100:.2f}%")
        except ValueError as e:
            return f"Error al entrenar el modelo: {str(e)}"
    else:
        return "El archivo no está disponible para el entrenamiento."


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model_filename = 'models/consumer_model.pkl'
    
    if not os.path.exists(model_filename):
        # Si el modelo no existe, mostrar mensaje de error y redirigir a la carga de datos
        return render_template('predict.html', error="El modelo no está entrenado. Serás redirigido para cargar un set de datos y entrenarlo.")

    if request.method == 'POST':
        try:
            # Cargar el modelo entrenado
            model = joblib.load(model_filename)

            # Recoger las respuestas del formulario
            features = [request.form['q1'], request.form['q2'], request.form['q3'],
                        request.form['q4'], request.form['q5'], request.form['q6'],
                        request.form['q7']]

            # Convertir las características a formato numérico
            features = [[float(x) for x in features]]

            # Realizar la predicción
            prediction = model.predict(features)

            # Obtener el tipo de consumidor a partir de la predicción
            consumer_type = consumer_types.get(prediction[0], "Tipo de consumidor no reconocido")

            return render_template('predict.html', result=f"Eres un {consumer_type}")
        except Exception as e:
            return f"Error al hacer la predicción: {str(e)}"

    return render_template('predict.html')



if __name__ == '__main__':
    app.run(debug=True)
