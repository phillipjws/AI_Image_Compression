import cv2
from deap import base, creator, tools, algorithms
import numpy as np
from PIL import Image
import sys


def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    return np.array(image)


def fitness_function(individual, image):
    pass


# Can probably use a library for this, not too sure how ATM
def calculate_quality(individual, image):
    pass


# Same as calculate quality
def calculate_file_size(individual, image):
    pass


def compress_image(individual, image):
    pass


# I think I know how to do all this, DEAP makes it pretty easy
def setup_genetic_algo(image):
    pass


def main(image_path):
    image = load_image(image_path)
    toolbox = setup_genetic_algo(image)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    main(image_path)
