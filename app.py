from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import uuid

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Ruta donde se almacenarán las gráficas generadas
GRAPHICS_FOLDER = 'static/graphics'

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

            return render_template('train_result.html', accuracy=f"Entrenamiento completo. Precisión del modelo: {accuracy * 100:.2f}%")
        except ValueError as e:
            return f"Error al entrenar el modelo: {str(e)}"
    else:
        return "El archivo no está disponible para el entrenamiento."

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model_filename = 'models/consumer_model.pkl'
    
    if not os.path.exists(model_filename):
        return render_template('predict.html', error="El modelo no está entrenado. Serás redirigido para cargar un set de datos y entrenarlo.")

    if request.method == 'POST':
        try:
            model = joblib.load(model_filename)
            features = [request.form['q1'], request.form['q2'], request.form['q3'],
                        request.form['q4'], request.form['q5'], request.form['q6'],
                        request.form['q7']]
            features = [[float(x) for x in features]]
            prediction = model.predict(features)
            consumer_type = consumer_types.get(prediction[0], "Tipo de consumidor no reconocido")
            return render_template('predict.html', result=f"Eres un {consumer_type}")
        except Exception as e:
            return f"Error al hacer la predicción: {str(e)}"
    return render_template('predict.html')

# Nueva ruta para generar gráfica K-Means
@app.route('/generate_kmeans', methods=['GET', 'POST'])
def generate_kmeans():
    file_path = session.get('filepath')
    if not file_path or not os.path.exists(file_path):
        return render_template('kmeans.html', error="No se ha cargado ningún set de datos para el entrenamiento.")

    if request.method == 'POST':
        df = pd.read_csv(file_path)
        X = df.iloc[:, 1:]  
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.dropna()

        if X.empty:
            return "No hay suficientes datos numéricos para aplicar K-Means."

        # Aplicamos K-Means con n clusters igual al número de columnas de X
        kmeans = KMeans(n_clusters=X.shape[1], random_state=42)
        kmeans.fit(X)

        df['cluster'] = kmeans.labels_

        # Aplicamos PCA para visualizar en 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Guardar gráfica en una carpeta con nombre único
        if not os.path.exists(GRAPHICS_FOLDER):
            os.makedirs(GRAPHICS_FOLDER)

        # Generar un nombre único para la gráfica
        graph_name = f"kmeans_{uuid.uuid4()}.png"
        graph_path = os.path.join(GRAPHICS_FOLDER, graph_name)

        # Generar la gráfica
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='viridis')
        plt.title(f"K-Means Clustering con {X.shape[1]} clústeres")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.savefig(graph_path)
        plt.close()

        return render_template('kmeans.html', graph_path=graph_path)

    return render_template('kmeans.html')

# Nueva ruta para listar las gráficas generadas
@app.route('/list_graphics', methods=['GET'])
def list_graphics():
    graphics = os.listdir(GRAPHICS_FOLDER)
    return render_template('list_graphics.html', graphics=graphics)

# Nueva ruta para generar PDF con las gráficas seleccionadas
@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    selected_graphics = request.form.getlist('graphics')
    pdf_path = os.path.join(GRAPHICS_FOLDER, f"exported_{uuid.uuid4()}.pdf")

    # Crear el archivo PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.drawString(100, height - 50, "Reporte de Gráficas Generadas")

    for graphic in selected_graphics:
        graphic_path = os.path.join(GRAPHICS_FOLDER, graphic)
        if os.path.exists(graphic_path):
            c.drawImage(graphic_path, 100, height - 300, width=400, height=200)
            c.showPage()

    c.save()

    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
