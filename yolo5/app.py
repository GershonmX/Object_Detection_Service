import time
from pathlib import Path
from flask import Flask, request
from detect import run
import uuid
import yaml
from loguru import logger
import os
import boto3
from pymongo import MongoClient


# Specify the bucket name
bucket_name = os.environ['BUCKET_NAME']

# Specify the key (path) of the image you want to download
image_key = img_name  # Assuming 'img_name' contains the image key

# Specify the local path where you want to save the downloaded image
local_image_path = f'static/data/{prediction_id}/{image_key}'

# Initialize Flask app
app = Flask(__name__)

# Load COCO names
with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

# Initialize S3 client
s3 = boto3.client('s3')

# Initialize MongoDB client
client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
db = client['your_database_name']  # Replace with your database name
collection = db['prediction_summaries']  # Replace with your collection name

@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())
     logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    img_name = request.args.get('imgName')

    # Specify the bucket name
    bucket_name = os.environ['BUCKET_NAME']

    # Specify the local path to save the downloaded image
    local_image_path = f'static/data/{prediction_id}/{img_name}'

    # Download the image from S3
    s3.download_file(bucket_name, img_name, local_image_path)
    logger.info(f'prediction: {prediction_id}/{local_image_path}. Download img completed')

    # Download the image from S3
    s3.download_file(bucket_name, img_name, local_image_path)
    logger.info(f'prediction: {prediction_id}/{local_image_path}. Download img completed')


# TODO download img_name from S3, store the local image path in original_img_path
    #  The bucket name should be provided as an env var BUCKET_NAME.
    original_img_path = ...

    logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).

    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': predicted_img_path,
            'labels': labels,
            'time': time.time()
        }

        # TODO store the prediction_summary in MongoDB

        return prediction_summary
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
