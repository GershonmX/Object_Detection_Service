import flask
from flask import request
import os
import requests
from bot import ImageProcessingBot

app = flask.Flask(__name__)

TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
if os.environ.get('TELEGRAM_APP_URL'):
    TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']
else:
    TELEGRAM_APP_URL = requests.get('http://169.254.169.254/latest/meta-data/public-ipv4').text


@app.route('/', methods=['GET'])
def index():
    return 'Ok'


@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


if __name__ == "__main__":
    bot = ImageProcessingBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL)
    app.run(host='0.0.0.0', port=8443)
