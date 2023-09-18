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

# Initialize S3 client
s3 = boto3.client('s3')

# Initialize Flask app
app = Flask(__name__)

# Initialize MongoDB client (replace with your MongoDB connection details)
client = MongoClient('mongodb://localhost:27017/')
db = client['mongodb']
collection = db['myReplicaSet']  # Use your collection name here

# Load COCO names
with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request.
    # This id can be used as a reference in logs to identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())
    logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    img_name = request.args.get('imgName')

    # Specify the local path to save the downloaded image

    img_path =img_name.split('/')[-1]
    local_dir = 'photos/'
    local_image_path = os.path.join(local_dir, img_path)  # Concatenate local_dir with img_path
    os.makedirs(local_image_path, exist_ok=True)
    original_img_path = local_dir + img_path


    # TODO download img_name from S3, store the local image path in original_img_path
    # Download the image from S3
    try:
        s3.download_file(bucket_name, img_name, original_img_path)
        logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
    except Exception as e:
        logger.error(f'Error downloading image: {e}')

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path, # Use the correct COCO YAML path
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # This is the path for the predicted image with labels
    predicted_img_path = f'static/data/{prediction_id}/{img_name}'

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
            'predicted_img_path': str(predicted_img_path),
            'labels': labels,
            'time': time.time()
        }

        # TODO store the prediction_summary in MongoDB
        collection.insert_one(prediction_summary)

        return prediction_summary
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
