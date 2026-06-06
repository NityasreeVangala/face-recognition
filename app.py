from flask import Flask, render_template, request
from scripts_day1.utils import predict_faces
from PIL import ImageDraw
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            filename = img_file.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            img_file.save(path)

            img, predictions = predict_faces(path)

            # Draw results
            draw = ImageDraw.Draw(img)
            for face in predictions:
                box = face["box"]
                label = f"{face['name']} ({face['confidence']})"
                color = "green" if face['name'] != "Unknown" else "red"
                draw.rectangle(box, outline=color, width=2)
                draw.text((box[0], box[1]-10), label, fill="white")

            result_path = os.path.join(UPLOAD_FOLDER, "result_" + filename)
            img.save(result_path)

            return render_template("index.html", image="result_" + filename, predictions=predictions)
    return render_template("index.html")

if __name__ == '__main__':
    print("🚀 Starting Flask App...")
    app.run(debug=True)
