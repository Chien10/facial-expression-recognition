import json, os, copy, logging
from pathlib import Path

from flask import Flask, request, redirect, render_template, jsonify, url_for

from fer.utils import *

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / 'ui'

# An instance of Flask class used as WSGI application
app = Flask(__name__, template_folder = str(WEB / 'templates'), static_folder = str(WEB / 'static'))
app.config['ALLOWED_IMAGE_EXTENSIONS'] = allowed_image_extensions
# Save all the uploaded images and detected faces to static folder
app.config['IMAGE_UPLOADS'] = str(WEB / 'static')
app.config['FACE_UPLOADS'] = str(WEB / 'static')

logging.basicConfig(level=logging.INFO)

@app.errorhandler(404)
def not_found():
    """Page not found"""
    return make_response(render_template("404.html"), 404)


@app.errorhandler(400)
def bad_request():
    """Bad request"""
    return make_response(render_template("400.html"), 400)


@app.errorhandler(500)
def server_error():
    """Internal server error"""
    return make_response(render_template("500.html"), 500)


# Method GET and POST will trigger our function
@app.route('/', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		
		if 'file' not in request.files:
			return redirect(request.url)

		file = request.files.get('file')

		if not file:
			return

		# Save uploaded image
		image_name = save_image(file=file, folder=app.config['IMAGE_UPLOADS'])
		
		# Make prediction
		#dir_path = os.path.dirname(os.path.realpath(app.config['IMAGE_UPLOADS']))
		image_path = os.path.join(app.config['IMAGE_UPLOADS'], image_name)
		emotion_names, prefixes = get_prediction(image_path, terminal=False)

		face_paths =  []
		for prefix in prefixes:
			face_path = prefix + '.jpg'
			face_paths.append(face_path)
		
		res = []
		for emotion_name, prefix in zip(emotion_names, prefixes):
			rec = {}
			rec['url'] = prefix + '.jpg'
			rec['label'] = emotion_name
			res.append(rec)

		return render_template('result.html', result=res)

	return render_template('index.html')

def main():
	app.run(host="0.0.0.0", debug=True, port=int(os.environ.get('PORT', 5000)))

if __name__ == '__main__':
	main()