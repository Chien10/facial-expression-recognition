import json, os, copy, logging
from pathlib import Path

from flask import Flask, request, redirect, render_template, jsonify, url_for

import fer.utils as utils

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / 'ui'

# An instance of Flask class used as WSGI application
app = Flask(__name__, template_folder = str(WEB / 'templates'), static_folder = str(WEB / 'static'))
app.config['ALLOWED_IMAGE_EXTENSIONS'] = allowed_image_extensions
# Save all the uploaded images and detected faces to static folder
app.config['IMAGE_UPLOADS'] = str(WEB / 'static')
app.config['FACE_UPLOADS'] = str(WEB / 'static')

logging.basicConfig(level=logging.INFO)

def lambda_handler(event, _context):
    """
    """
    image = _load_image(event)
    emotion_names, prefixes = utils.get_prediction(image, terminal = True)

    result = {}
    for emotion, prefix in zip(emotion_names, prefixes):
        print('Emotion of face {}: {}'.format(emotion, prefix))
        result[prefix] = emotion
    return result

def _load_image(event) -> Image:
    """
    """
    image_url = event.get("image_url")
    if image_url is None:
        return "No image url is found in events"
    print('IMAGE URL INFO: {}'.fomrat(image_url))
    return utils.read_image_pil(image_url, grayscale=True)