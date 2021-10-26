
from flask import Flask, render_template, request
from cv2 import cv2
from keras.models import load_model
import boto3
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime
import os


model = load_model('modelfl.h5')

class_names = ['daisy', 'dandelion','rose','sunflower','tulip']


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


now = datetime.now()
dt_string = str(now.strftime("%d/%m/%Y %H:%M:%S"))
ACCESS_KEY_ID = 'AKIA5YTDSEU7HOKXECX7'
ACCESS_SECRET_KEY = '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'
s3 = boto3.client('s3',
                  aws_access_key_id=ACCESS_KEY_ID,
                  aws_secret_access_key=ACCESS_SECRET_KEY,
                  region_name='us-east-2')
BUTKET_NAME = 'tinh0110s3'

dynamo_db = boto3.client('dynamodb', aws_access_key_id=ACCESS_KEY_ID,
                         aws_secret_access_key=ACCESS_SECRET_KEY,
                         region_name='us-east-2')


polly_client = boto3.client('polly',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_SECRET_KEY,
    region_name='us-east-2')

@app.route('/')
def index():
    return render_template('index.html', data="", class_names=enumerate(class_names), percents=None)


@app.route('/after', methods=['GET', 'POST'])
def after():
    
    global model, class_names,dynamo_db, s3
    PARTTION_KEY = 'predict'
    SORT_KEY = 'time'

    img = request.files['file_image']

    # if not img:
    #     return render_template('index.html', data="", class_names=enumerate(class_names), percents=None)

    img.save('static/image/file.jpg')

    if img:
        file_name = secure_filename(img.filename)
        img.save(img.filename)
        s3.upload_file(Bucket=BUTKET_NAME, Filename=file_name, Key=file_name)
    image = cv2.imread('static/image/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.reshape(image, (1, 224, 224, 3))
    prediction = model.predict(image)
    pred_labels = np.argmax(prediction, axis=1)
    prediction = [round(float(prediction[0][i])*100, 3)  # lấy %
                  for i in range(len(class_names))]
    final = class_names[int(pred_labels)].replace('_', ' ')

    item = {
        PARTTION_KEY: {
            "S": final
        },
        SORT_KEY: {
            "S": dt_string
        }
    }
    dynamo_db.put_item(
        TableName='tinh1001DynamoDB',
        Item=item
    )

    response = polly_client.synthesize_speech(VoiceId='Salli',
                                              OutputFormat='mp3',
                                              Text=final
                                              )
    file = open('static/speech.mp3', 'wb')
    file.write(response['AudioStream'].read())
    file.close()

    return render_template('index.html', data=final, class_names=enumerate(class_names), percents=prediction)


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=8080)
    app.run(debug=True)
