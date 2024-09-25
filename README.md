# Optic Classifier Modulation Using Fresnel Diffraction

## Overview

This project involves designing an optical system to classify digits using Fresnel diffraction. The system uses an input image of a digit from the MNIST dataset, which passes through a lens, mask, and finally onto an output screen. The mask is optimized using a genetic algorithm to enhance classification accuracy.

## Project Components

- **Input**: Digit images from the MNIST dataset.
- **Optical System**: Consists of a lens, mask, and output screen to classify the digits.
- **Optimization**: Uses genetic algorithms to optimize the mask based on light diffraction patterns.
- **Diffraction Analysis**: Utilizes Fresnel diffraction principles to manipulate light patterns for classification.

## Key Features

1. **Data Preprocessing**: 
   - Cleans and normalizes MNIST data.
   - Filters images to specific labels (e.g., 0, 1, 2, 3).

2. **Fresnel Diffraction**:
   - Simulates light propagation through the system using Fourier optics.
   - Applies a custom mask to classify the input images based on diffraction patterns.

3. **Fitness Functions**:
   - Evaluates the intensity of light fields in different quadrants to determine the classification result.
   - Optimizes the mask for accurate digit recognition.

4. **Genetic Algorithm**:
   - Initializes a population of masks.
   - Iteratively optimizes masks using selection, crossover, and mutation.
   - Trains over multiple generations to achieve the best classification accuracy.

## Usage

### Prerequisites
- Python 3.x
- Required libraries: `numpy`, `matplotlib`, `scipy`, `pandas`

### Running the Code

1. Clone the repository and navigate to the project directory.
2. Install the necessary dependencies using:
   ```bash
   pip install -r requirements.txt
