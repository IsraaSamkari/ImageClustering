from flask import Flask, request, jsonify, render_template_string
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

@app.route('/')
def home():
    return render_template_string(open('templates/index.html').read())

@app.route('/process', methods=['POST'])
def process_images():
    cluster_num = int(request.form['clusterNum'])
    images = request.files.getlist('images')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Save images and extract features
    image_paths = []
    features = []
    for img in images:
        filename = secure_filename(img.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(file_path)
        image_paths.append(file_path)
        features.append(extract_features(file_path))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=cluster_num, random_state=0)
    clusters = kmeans.fit_predict(features)
    
    # Generate output HTML
    output_html = ''
    for cluster in range(cluster_num):
        output_html += f"<h2>Cluster {cluster}</h2><div>"
        for i, label in enumerate(clusters):
            if label == cluster:
                output_html += f'<img src="/{image_paths[i]}" alt="Image" width="200">'
        output_html += '</div>'
    
    return jsonify({'html': output_html})

if __name__ == '__main__':
    app.run(debug=True)
