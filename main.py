import cv2
from deap import base, creator, tools, algorithms
import numpy as np
from PIL import Image
import sys
import os


def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    return np.array(image)


def calculate_mse(individual, image):
    # assuming individual and image are both numpy arrays
    # Calculate the Mean Squared Error of the two images
    squared_diff = (individual - image) ** 2
    mse = np.mean(squared_diff)

# I'm using peak signal to noise ratio for the fitness function
def fitness_function(individual, image, fitness):
    # image parameter will be the original image that im comparing the individual to.
    mse = calculate_mse(individual, image)

    MAX = 255.0  # Maximum pixel value for 8-bit images
    psnr = 10 * np.log10((MAX ** 2) / mse)

    if psnr > fitness:
        return False
    else:
        return True

# Can probably use a library for this, not too sure how ATM
def calculate_quality(individual, image):
    pass


def calculate_file_size(file_path):
    return os.path.getsize(file_path)

def compress_image(image, compression_quality):
    # Compress an image and save it to a temporary file to measure file size
    compressed_image_path = 'temp_compressed.jpg'
    cv2.imwrite(compressed_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality])
    return compressed_image_path


# I think I know how to do all this, DEAP makes it pretty easy
def setup_genetic_algo(image):
    pass


def save_best_compressed_image(best_individual, image_path):
    pass


def main(image_path):
    image = load_image(image_path)
    toolbox = setup_ga(image)
    population = toolbox.population(n=50)  # Adjust population size

    NGEN = 40  # Number of generations
    CXPB, MUTPB = 0.5, 0.2  # Crossover and mutation probabilities

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        fits = [ind.fitness.values[0] for ind in population]
        print(f"Generation {gen}, Best fitness: {max(fits)}")

    best_ind = tools.selBest(population, 1)[0]
    print("Best individual:", best_ind)

    # Save the best compressed image
    save_compressed_image(image_path, best_ind)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    main(image_path)
