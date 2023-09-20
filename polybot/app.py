import flask
from flask import request
import os
import requests
from bot import Bot, QuoteBot, ImageProcessingBot

app = flask.Flask(__name__)

TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']

# Load AWS credentials from /root/.aws/credentials
#aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
#aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')


@app.route('/', methods=['GET'])
def index():
    return 'Ok'


@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


if __name__ == "__main__":
    #bot = QuoteBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL)
    #bot = QuoteBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL)
    #bot = Bot(TELEGRAM_TOKEN, TELEGRAM_APP_URL)
    bot = ImageProcessingBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL)

    app.run(host='0.0.0.0', port=8443)
