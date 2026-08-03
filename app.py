from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the trained model
model = load_model("./model/CNN_Model.h5")

# Define emotion classes
Emotion_Classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the music player data
Music_Player = pd.read_csv("./dataset/data_moods.csv")

# Preprocess the image
def load_and_prep_image(filename, img_shape=48):   

    img = cv2.imread(filename)

    GrayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier("./haar/haarcascade_frontalface_default.xml") 
    
    faces = faceCascade.detectMultiScale(GrayImg, 1.1, 4)
    
    for x,y,w,h in faces:
        
        roi_GrayImg = GrayImg[ y: y + h , x: x + w ]
        roi_Img = img[ y: y + h , x: x + w ]
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        
        faces = faceCascade.detectMultiScale(roi_Img, 1.1, 4)
        
        if len(faces) == 0:
            print("No Faces Detected")
        else:
            for (ex, ey, ew, eh) in faces:
                img = roi_Img[ ey: ey+eh , ex: ex+ew ]
    
    RGBImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    RGBImg= cv2.resize(RGBImg,(img_shape,img_shape))

    RGBImg = RGBImg/255.

    return RGBImg

# Predict emotion and recommend songs
def predict_emotion_and_recommend(filename):
    img = load_and_prep_image(filename)
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = Emotion_Classes[np.argmax(pred)]
    recommendations = recommend_songs(pred_class)
    return pred_class, recommendations

# Recommend songs based on predicted emotion
def recommend_songs(pred_class):
    recommendations = []

    if pred_class == 'Disgust':
        recommendations = get_music_recommendations('Sad')
    elif pred_class in ['Happy', 'Sad']:
        recommendations = get_music_recommendations('Happy')
    elif pred_class in ['Fear', 'Angry']:
        recommendations = get_music_recommendations('Calm')
    elif pred_class in ['Surprise', 'Neutral']:
        recommendations = get_music_recommendations('Energetic')

    return recommendations

# Get music recommendations for a given mood
def get_music_recommendations(mood):
    songs = Music_Player[Music_Player['mood'] == mood]
    songs = songs.sort_values(by="popularity", ascending=False)[:5]
    recommendations = songs[['album', 'artist', 'name', 'popularity', 'release_date']].to_dict(orient='records')
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')
# API endpoint for prediction and recommendations
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join('./uploads', filename)
    file.save(file_path)

    emotion, recommendations = predict_emotion_and_recommend(file_path)

    return jsonify({'emotion': emotion, 'recommendations': recommendations}), 200

if __name__ == '__main__':
    app.run(debug=True,port=8080)
