# ECE470Project

# Image Compression using Genetic Algorithms

This project aims to reduce the file size of high-resolution images while maintaining visual quality using Genetic Algorithms (GA). By optimizing the compression parameters, we can achieve efficient storage and transmission of images.

## Problem Statement and Motivation

### Problem
High-resolution images consume significant storage space and bandwidth. Traditional methods for reducing file size often lead to a loss in image quality. This project uses Genetic Algorithms to find the optimal balance between file size and image quality.

### Importance
- Reduces storage costs.
- Saves bandwidth during transmission.
- Enhances user experience with faster load and download times.

### Why AI Techniques
Genetic Algorithms can explore large and complex search spaces, making them capable of finding the best solutions quickly and efficiently.

## Tasks and Workflow

### Task 1: Problem Definition and Setup
- **Objective**: Define the problem and set up the environment.
- **Steps**:
  1. Define the objective function to minimize file size while maintaining visual quality.
  2. Set up the required packages and dependencies.

### Task 2: Load and Preprocess Image
- **Objective**: Load and prepare the image for processing.
- **Steps**:
  1. Load the high-resolution image.
  2. Convert the image to the RGB format if necessary.

### Task 3: Define Fitness Function
- **Objective**: Create a function to evaluate the quality and size of the compressed image.
- **Steps**:
  1. Implement the fitness function to balance image quality and file size.
  2. Define quality and size calculation methods.

### Task 4: Initialize Genetic Algorithm
- **Objective**: Set up the Genetic Algorithm parameters and initial population.
- **Steps**:
  1. Define the initial population.
  2. Set parameters such as population size and mutation rate.

### Task 5: GA Operations
- **Objective**: Implement genetic operations to evolve the population.
- **Steps**:
  1. Implement selection, crossover, and mutation operations.
  2. Apply these operations to generate new populations.

### Task 6: Evaluate and Select
- **Objective**: Evaluate the fitness of individuals and select the best ones.
- **Steps**:
  1. Calculate the fitness of each individual.
  2. Select the best individuals for the next generation.

### Task 7: Iterate
- **Objective**: Iterate the GA process to find the optimal solution.
- **Steps**:
  1. Repeat the genetic operations for a fixed number of generations or until convergence.

### Task 8: Post-process and Save Image
- **Objective**: Apply the best compression settings and save the compressed image.
- **Steps**:
  1. Use the best solution to compress the image.
  2. Save the compressed image.

## Installation

To run this project, you'll need to install the required packages. You can do this by running the following command:

```bash
pip install -r requirements.txt
