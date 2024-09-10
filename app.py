from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            # Asegúrate de que el directorio existe
            temp_directory = os.path.join('static', 'temp')
            if not os.path.exists(temp_directory):
                os.makedirs(temp_directory)  # Crea el directorio si no existe

            filepath = os.path.join(temp_directory, file.filename)
            file.save(filepath)
            session['filepath'] = filepath  # Guarda la ruta del archivo en la sesión
            data = pd.read_csv(filepath)
            return render_template('upload.html', data=data.to_html())
    return render_template('upload.html')

@app.route('/train', methods=['GET'])
def train():
    # Obtener la ruta del archivo subido desde la sesión
    file_path = session.get('filepath')
    if file_path and os.path.exists(file_path):
        # Cargar los datos del CSV
        df = pd.read_csv(file_path)

        # Excluir la columna de nombres, asumiendo que es la primera columna
        X = df.iloc[:, 1:]  # Excluye la primera columna (los nombres)
        y = df.iloc[:, 1]  # Suponemos que la columna objetivo es la primera pregunta (puedes ajustar esto según tu lógica)

        # Convertir todas las columnas restantes a números
        X = X.apply(pd.to_numeric, errors='coerce')

        # Eliminar filas con valores NaN que pueden haber sido creados por la conversión
        X = X.dropna()

        # Verificar que queden datos para entrenar
        if X.empty or y.empty:
            return "No hay suficientes datos numéricos para entrenar el modelo."

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            model_filename = 'models/consumer_model.pkl'
            if not os.path.exists('models'):
                os.makedirs('models')  # Crea la carpeta models si no existe
            joblib.dump(model, model_filename)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            return f"Entrenamiento completo. Precisión del modelo: {accuracy * 100:.2f}%"
        except ValueError as e:
            return f"Error al entrenar el modelo: {str(e)}"
    else:
        return "El archivo no está disponible para el entrenamiento. Asegúrate de haber cargado un archivo válido."

if __name__ == '__main__':
    app.run(debug=True)
