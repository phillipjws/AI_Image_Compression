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
    file_size = os.path.getsize(image_path)
    return np.array(image), file_size


def calculate_mse(imageA, imageB):
    if imageA.shape != imageB.shape:
        imageA = cv2.resize(imageA, (imageB.shape[1], imageB.shape[0]))
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def fitness_function(individual, image, original_file_size):
    compression_quality = int(individual[0] * 100)
    resolution_scale = individual[1]
    color_depth = int(individual[2] * 256)
    compressed_image_path = compress_image(image, compression_quality, resolution_scale, color_depth)
    compressed_image, _ = load_image(compressed_image_path)
    mse = calculate_mse(compressed_image, image)
    psnr = 10 * np.log10((255.0 ** 2) / mse)
    file_size = os.path.getsize(compressed_image_path)

    size_penalty = 0 if file_size < original_file_size else (file_size - original_file_size)

    return (psnr, -file_size - size_penalty)  # Adding size penalty


def compress_image(image, compression_quality, resolution_scale, color_depth):
    if color_depth == 0:
        color_depth = 1

    height, width, channels = image.shape
    new_width = int(width * resolution_scale)
    new_height = int(height * resolution_scale)
    new_width = max(new_width, 1)
    new_height = max(new_height, 1)
    resized_image = cv2.resize(image, (new_width, new_height))

    scale_factor = 256 // color_depth
    quantized_image = (resized_image // scale_factor) * scale_factor
    quantized_image = np.clip(quantized_image, 0, 255)
    pil_image = Image.fromarray(quantized_image.astype('uint8'), 'RGB')

    compressed_image_path = 'temp_compressed.jpg'
    pil_image.save(compressed_image_path, quality=compression_quality, format='JPEG')
    return compressed_image_path


def save_best_compressed_image(best_individual, image_path):
    original_image, _ = load_image(image_path)
    
    compression_quality = int(best_individual[0] * 100)
    resolution_scale = best_individual[1]
    color_depth = int(best_individual[2] * 256)

    compressed_image_path = compress_image(original_image, compression_quality, resolution_scale, color_depth)
    
    output_path = "compressed_" + image_path.split('/')[-1]
    os.rename(compressed_image_path, output_path)
    
    print(f"Best compressed image saved as {output_path}")


def log_fitness(gen, best_fitness, avg_fitness, image_path):
    base, _ = os.path.splitext(image_path)
    with open(f"{base}_fitness_log.txt", "a") as log_file:
        log_file.write(f"Generation {gen}, Best fitness: {best_fitness}, Average fitness: {avg_fitness}\n")


def setup_genetic_algo(image_path):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("attr_scale", random.uniform, 0.5, 1)
    toolbox.register("attr_depth", random.uniform, 8/256, 1 - 1e-10)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_float, toolbox.attr_scale, toolbox.attr_depth), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Load the image once and use it for evaluating individuals
    image = load_image(image_path)
    toolbox.register("evaluate", fitness_function, image=image)
    
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    return toolbox


def main(image_path):
    toolbox = setup_genetic_algo(image_path)

    original_image, original_file_size = load_image(image_path)
    toolbox.register("evaluate", fitness_function, image=original_image, original_file_size=original_file_size)

<<<<<<< Updated upstream
    population_n = 300
    population = toolbox.population(n=population_n)

    NGEN = 400
=======
    population_n = 700
    population = toolbox.population(n=population_n)

    NGEN = 700
>>>>>>> Stashed changes
    CXPB, MUTPB = 0.7, 0.3

    base, _ = os.path.splitext(image_path)
    with open(f"{base}_fitness_log.txt", "a") as log_file:
        log_file.write(f"Population: {population_n}, Number of Generations: {NGEN}\n")

    best_fitness_previous = float('-inf')

    for gen in range(NGEN):
        # Ensure all individuals are evaluated
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        valid_fitnesses = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
        avg_fitness = np.mean(valid_fitnesses) if valid_fitnesses else 0
        
        population = toolbox.select(offspring, k=len(population))

        best_fitness = max(valid_fitnesses) if valid_fitnesses else 0
        print(f"Generation {gen}, Best fitness: {best_fitness}, Average fitness: {avg_fitness}")
        log_fitness(gen, best_fitness, avg_fitness, image_path)

        if best_fitness > best_fitness_previous:
            best_fitness_previous = best_fitness
            MUTPB = max(0.1, MUTPB - 0.01)  # Decrease mutation rate gradually
        else:
            MUTPB = min(0.3, MUTPB + 0.02)  # Increase mutation rate if no improvement

    best_ind = tools.selBest(population, 1)[0]
    print("Best individual:", best_ind)
    save_best_compressed_image(best_ind, image_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    main(image_path)
