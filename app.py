from flask import Flask, render_template, request
from models import MobileNet
import os
from math import floor
from werkzeug.utils import secure_filename
import io
import base64
from PIL import Image

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

model = MobileNet()
allowed_extensions = {'png','jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/infer', methods=['POST'])
def success():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        file_names = []
        pred_list  = []
        img_data_list = []
        for file in files:
            if file and allowed_file(file.filename):
                saveLocation = file.filename
                file.save(saveLocation)

                inference, confidence = model.infer(saveLocation)
                confidence = floor(confidence * 10_000) / 100

                file_names.append(secure_filename(saveLocation))

                tup = (inference, confidence)
                pred_list.append(tup)

                im = Image.open(saveLocation)
                data = io.BytesIO()
                im.save(data, 'JPEG')
                encoded_img_data = base64.b64encode(data.getvalue())
                img_data_list.append(encoded_img_data.decode('utf-8'))
                os.remove(saveLocation)

        return render_template('inference.html', predictions=pred_list, img_data=img_data_list, length=len(pred_list))

if __name__ == '__main__':
    print('cwd:', os.getcwd())
    app.debug = True
    port = int(os.environ.get("PORT",5000)) # Use heroku supplied port else 5000
    app.run(host='0.0.0.0', port=port, debug=True)

