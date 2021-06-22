import argparse
from PIL import Image

from utils import *

def predict(image_file: str):
	dir_path = os.path.dirname(os.path.realpath(image_file))
	img_path = os.path.join(dir_path, image_file.split('/')[1])
	
	image = Image.open(img_path)
	emotion_name = get_prediction(image, terminal=True)
	return emotion_name

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_file', help='the image file name to read and the folder containing it',
						type=str)
	args = parser.parse_args()

	image_file = args.image_file
	emotion_name = predict(image_file)
	print('Emotion:', emotion_name)