from flask import Flask, render_template, request
from models import MobileNet
import os
from math import floor
from werkzeug.utils import secure_filename
import io
import base64
import sys
from PIL import Image
import pandas as pd
from datetime import datetime

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './static/uploads/'    # To host static images
app.config['inference_log'] = os.path.join(app.config['UPLOAD_FOLDER'], 'inference_log.csv')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Limit the memory to 10 MB for security reasons

model = MobileNet()
allowed_extensions = {'png','jpg', 'jpeg'}
result_df_column_names = ['Inference', 'Confidence', 'Time_stamp']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    #file_name = 'sample_upload_img.jpg'
    file_name = app.config['sample_img']
    file_infer = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    inference, confidence = model.infer(file_infer)
    confidence = floor(confidence * 10_000) / 100
    result = str(inference) + ", Confidence: " + str(confidence)
    return render_template('index.html', sample_result=result)

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

        if len(files) > 3:
            msg = 'Uploaded ' + str(len(files)) + ' files. You can upload only upto 3 files in jpg or jpeg formats.'
            return render_template('inference.html', error_msg=msg)

        for file in files:
            if allowed_file(file.filename):
                continue
            else:
                msg = str(file.filename) + ' file that you are trying to upload has incorrect file format. Please upload jpg or jpeg formats only.'
                return render_template('inference.html', error_msg=msg)

        if os.path.isfile(app.config['inference_log']):
            latest_5 = get_latest_5()
        else:
            df = pd.DataFrame(columns = result_df_column_names)
            df.to_csv(app.config['inference_log'],index=False)
            latest_5 = get_latest_5()

        for file in files:
            if file and allowed_file(file.filename):
                saveLocation = file.filename
                file.save(saveLocation)

                inference, confidence = model.infer(saveLocation)
                confidence = floor(confidence * 10_000) / 100

                file_names.append(secure_filename(saveLocation))

                time_stamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                tup = (inference, confidence, time_stamp)
                pred_list.append(tup)

                im = Image.open(saveLocation)
                data = io.BytesIO()
                im.save(data, 'JPEG')
                encoded_img_data = base64.b64encode(data.getvalue())
                img_data_list.append(encoded_img_data.decode('utf-8'))
                os.remove(saveLocation)

        save_inference_log(pred_list)

        return render_template('inference.html', predictions=pred_list, img_data=img_data_list, length=len(pred_list),
                                prev_results=latest_5, enumerate=enumerate)

def get_latest_5():
    df = pd.read_csv(app.config['inference_log'])
    df.sort_values(by=['Time_stamp'], inplace=True, ascending=False)
    latest_5 = df.head(5)
    return latest_5.values.tolist()

def save_inference_log(pred_list):
    df = pd.read_csv(app.config['inference_log'])
    df = df.append(pd.DataFrame(pred_list, columns=result_df_column_names),ignore_index=True)
    df.to_csv(app.config['inference_log'],index=False)

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT",80)) # Use heroku supplied port else 5000
    app.config['sample_img'] = sys.argv[1]
    app.run(host='0.0.0.0', port=port, debug=True)

