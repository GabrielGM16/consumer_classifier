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
            session['filepath'] = filepath
            data = pd.read_csv(filepath)
            return render_template('upload.html', data=data.to_html())
    return render_template('upload.html')

@app.route('/train', methods=['GET'])
def train():
    if 'filepath' in session:
        filepath = session['filepath']
        if os.path.exists(filepath):
            data = pd.read_csv(filepath)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            model_filename = 'models/consumer_model.pkl'
            joblib.dump(model, model_filename)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            return f"Entrenamiento completo. Precisión del modelo: {accuracy * 100:.2f}%"
        else:
            return "El archivo ya no está disponible para el entrenamiento."
    else:
        return "No se ha subido ningún archivo. Por favor, sube un archivo para entrenar el modelo."

if __name__ == '__main__':
    app.run(debug=True)
