# ECE470Project

## Installation

To run this project, you'll need to install the required packages. You can do this by running the following command:

```bash
pip install -r requirements.txt
```
## Running The Program

To run the project, ensure you have the required packages installed. 

You should then be able to run it with:
```
python3 main.py <image.jpg>
```

where image.jpg is whatever image you want to run it with.

It will save a log under the name 'fitness_log.txt' and save the image as 'compressed_<image>.jpg' where <image> is the name you chose initially

If you want it to run faster, you can reduce the population, or the number of generations. 

To do this you can change the variables population, and NGEN, in line 100, and 103 of main.py

As a baseline, population should probably be greater than 100, and number of generations should probably be more than 50

It may take a long time to run, depending on the speed of your machine you may want to adjust these accordingly, it will print the best fitness and average fitness of each generation after each iteration, so you can judge how long each iteration takes. 