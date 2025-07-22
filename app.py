from flask import Flask, render_template, request
import os
import json
import torch
import joblib
from torchvision import models
from werkzeug.utils import secure_filename
from utils.video_utils import extract_frames
from utils.extract_features import extract_face_features

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4'}
classifier = joblib.load('model/classifier.pkl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet18 = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1]).to(device)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files.get('video')

        # T004 - No file uploaded
        if not video or video.filename == '':
            return render_template('index.html', error="No file selected.")

        # T003 - Unsupported file type
        if not allowed_file(video.filename):
            return render_template('index.html', error="Unsupported file type.")

        filename = secure_filename(video.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        video.save(filepath)

        try:
            with open('model/metrics.json', 'r') as f:
                metrics = json.load(f)
        except FileNotFoundError:
            metrics = {"val_accuracy": "N/A", "val_f1": "N/A"}

        predictions = []
        try:
            frames = extract_frames(filepath)

            for frame in frames[:5]:  # Efficiency limit
                try:
                    feature = extract_face_features(frame, feature_extractor, device)
                    pred = classifier.predict([feature])[0]
                    predictions.append(pred)
                except Exception as e:
                    print(f"[!] Frame skipped: {e}")
        except Exception as e:
            print(f"[!] Video processing failed: {e}")
            # T005 - Corrupted file
            return render_template('index.html', error="Error processing video. It may be corrupted.")

        # T005 - No usable frames
        if not predictions:
            return render_template('index.html', error="Could not process any frames. Try another video.")

        final_prediction = round(sum(predictions) / len(predictions))
        confidence = round(predictions.count(final_prediction) / len(predictions) * 100, 2)
        label = "Real" if final_prediction == 1 else "Fake"

        return render_template('result.html', prediction=label, confidence=confidence, metrics=metrics)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
