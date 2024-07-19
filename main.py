import random
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
    return mse


def fitness_function(individual, image):
    # Compress the image and evaluate its fitness based on PSNR
    compressed_image_path = compress_image(image, int(individual[0] * 100))
    compressed_image = load_image(compressed_image_path)
    
    mse = calculate_mse(compressed_image, image)
    MAX = 255.0  # Maximum pixel value for 8-bit images
    psnr = 10 * np.log10((MAX ** 2) / mse)
    
    return psnr,


def compress_image(image, compression_quality):
    # Compress an image and save it to a temporary file to measure file size
    compressed_image_path = 'temp_compressed.jpg'
    cv2.imwrite(compressed_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality])
    return compressed_image_path


def setup_genetic_algo(image):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function, image=image)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def save_best_compressed_image(best_individual, image_path):
    original_image = load_image(image_path)
    compressed_image_path = compress_image(original_image, int(best_individual[0] * 100))
    
    output_path = "compressed_" + image_path.split('/')[-1]
    os.rename(compressed_image_path, output_path)
    
    print(f"Best compressed image saved as {output_path}")


def main(image_path):
    image = load_image(image_path)
    toolbox = setup_genetic_algo(image)
    population = toolbox.population(n=50)  # Population size

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
    save_best_compressed_image(best_ind, image_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    main(image_path)
