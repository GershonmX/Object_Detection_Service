import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import requests
import boto3

class Bot:

    def __init__(self, token, telegram_chat_url):
        # Load AWS credentials from /root/.aws/credentials
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        # Configure boto3 with AWS credentials
        self.s3 = boto3.client('s3', 'aws_access_key_id', 'aws_secret_access_key')

        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):
        def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)

            # TODO upload the photo to S3
            s3_bucket_name = 'gershonm-s3'
            s3_photo_key = f'photos/{photo_name}'
            self.s3.upload_file(photo_path, s3_bucket_name, s3_photo_key)
            logger.info(f'Uploaded user\'s photo to S3: {s3_photo_key}')

            # TODO send a request to the `yolo5` service for prediction
            yolo5_service_url = 'http://localhost:8081/predict'  # Replace with actual URL
            response = requests.post(yolo5_service_url, params={'imgName': s3_photo_key})

            # TODO send results to the Telegram end-user
            prediction_results = response.json()

            # Send the results back to the Telegram end-user
            self.send_text(msg['chat']['id'], f'Object detection results: {prediction_results}')

