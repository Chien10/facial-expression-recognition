import io, json, os
from datetime import datetime
from pathlib import Path

from joblib import load
import smart_open

from PIL import Image
from skimage.feature import hog
import numpy as np
import cv2

from typing import Union

from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
FER_MODELS_DIRNAME = ROOT / "fer_models"
FD_MODELS_DIRNAME = ROOT / "pretrained_face_detection_models"
CLASSES_DIRNAME = ROOT / "classes"
FACE_UPLOADS = ROOT.parent / 'ui' / 'static'

################################################### SVM Model #######################################################################
allowed_image_extensions = ['JPG']
#face_uploads = os.path.join('../ui/static/', 'upload_faces/')

def load_fer_model(path: str):
	"""
		Returns pre-trained model in scikit-learn
		Argument:
				path (str): path to the pre-trained model
		Returns:
				model: scikit-learn model
	"""
	model = load(path)
	print(type(model))
	return model

def load_face_detection_model(model_type: str):
	if model_type == 'haar':
		dir_path = os.path.dirname(os.path.realpath(cv2.data.haarcascades + \
								'pretrained_face_detection_models/haarcascade_frontalface_default.xml'))
		model_path = os.path.join(dir_path, cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
		#model_path = os.path.dirname(
		#	Path(os.path.realpath(cv2.data.haarcascades)) / FD_MODELS_DIRNAME / 'haarcascade_frontalface_default.xml'
		#)
		face_cascade = cv2.CascadeClassifier(model_path)

		return face_cascade
	elif model_type == 'facenet':
		pass
	else:
		print("'model_type' argument was not valid")
		return None

def extract_hog_features(image) -> tuple:
	"""
		Extract Histogram of Oriented Gradients Features from 
		a given image
		Argument:
				image (): an image to extract
		Returns:
				fd: extracted feature
				hog_image: image for visualization (doesn't make any sense!)
	"""
	ppc = 4
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(ppc, ppc),
                      cells_per_block=(1, 1), block_norm= 'L2', visualize=True)

	return fd, hog_image

def preprocess_image(image, terminal=False):
	"""
		Preprocess the image before applying prediction
		Arguments:
					image (): the image to be preprocessed
					terminal (): using in terminal or web app
		Returns:
					new_image: preprocessed image
	"""

	if not terminal:
		pil_image = Image.open(image)
	else:
		pil_image = image

	# Model to detect face
	face_detection_model = load_face_detection_model(model_type='haar')
	image = np.array(pil_image)
	# Convert image to gray-level
	if len(image.shape) > 2:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		gray = image
	# Detect face
	faces = face_detection_model.detectMultiScale(gray, 1.3, 5)

	if faces == ():
		return None, None

	# Assume that we have only one face
	"""
	x, y, w, h = face[0]
	face = image[y: y + h, x: x + w]
	# Save detected face to folder #
	save_image = Image.fromarray(face)
	# Name of the detected face
	prefix = datetime.now().strftime('%Y-%B-%d-%H_%M_%S_%f')
	dir_path = os.path.dirname(os.path.realpath(face_uploads))
	model_path = os.path.join(dir_path, prefix + '.jpg')
	save_image.save(model_path)

	face = cv2.resize(face, (48, 48))

	fd, _ = extract_hog_features(face)
	new_image = fd

	new_image = fd.reshape(1, -1)
	"""
	# Handle multiple faces
	prefixes = []
	new_images = []

	for face in faces:
		x, y, w, h = face
		face = image[y: y + h, x: x + w]
		save_image = Image.fromarray(face)

		prefix = datetime.now().strftime('%Y-%B-%d-%H_%M_%S_%f')
		#dir_path = os.path.dirname(os.path.realpath(face_uploads))
		model_path = os.path.join(str(FACE_UPLOADS), prefix + '.jpg')
		save_image.save(model_path)

		face = cv2.resize(face, (48, 48))

		fd, _ = extract_hog_features(face)
		new_image = fd.reshape(1, -1)
		
		new_images.append(new_image)
		prefixes.append(prefix)

	return new_images, prefixes

def get_prediction(image, terminal=False) -> tuple:
	"""
		Apply a pretrained model to a given image
		Arguments:
					image ():
					terminal ():
		Returns:
					emotion_class (str):
	"""

	### Load SVM model ###
	#dir_path = os.path.dirname(os.path.realpath('fer_models/ovo_hog_4x4_svm.joblib'))
	#model_path = os.path.join(dir_path, 'ovo_hog_4x4_svm.joblib')
	model_path = FER_MODELS_DIRNAME / 'ovo_hog_4x4_svm.joblib'
	model = load_fer_model(model_path)

	### Load VGG16 model ###
	

	#dir_path = os.path.dirname(os.path.realpath('classes.json'))
	#json_path = os.path.join(dir_path, 'classes.json')
	json_path = CLASSES_DIRNAME / 'classes.json'
	with open(json_path, encoding='utf-8') as f:
		emotions = json.load(f)

	images, prefixes = preprocess_image(image, terminal=terminal)
	if images is None:
		return 'Cannot recognize face to detect emotion!', prefixes

	emotion_classes = []
	for image in images:
		emotion_id = model.predict(image)[0]
		emotion_id = str(emotion_id)

		emotion_class = emotions[emotion_id]
		emotion_classes.append(emotion_class)

	return emotion_classes, prefixes

def is_image_valid(image_name: str) -> bool:
	if not '.' in image_name:
		return False

	img_extension = image_name.rsplit('.', 1)[1]
	if img_extension.upper() in allowed_image_extensions:
		return True

	return False

def save_image(file, folder: str):
	img_name = file.filename

	if is_image_valid(img_name):
		filename = secure_filename(img_name)
		#dir_path = os.path.dirname(os.path.realpath(os.path.join(folder)))
		save_path = os.path.join(folder, img_name)
		file.save(save_path)

		return img_name

	return None

def read_image_pil_file(image_file, grayscale: bool = False) -> Image:
	with Image.open(image_file) as image:
		if grayscale:
			image = image.convert(mode='L')
		else:
			image = image.convert(mode=image.mode)
		return image

def read_image_pil(image_uri: Union[Path, str], grayscale: bool = False) -> Image:
	with smart_open.open(image_uri, "rb") as image_file:
		return read_image_pil_file(image_file, grayscale)

################################################### VGG16 Model #######################################################################
