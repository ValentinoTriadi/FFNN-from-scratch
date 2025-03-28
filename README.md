# FFNN-from-scratch

> **Tugas Besar IF3270 - Machine Learning**

> A Feedforward Neural Network (FFNN) is a powerful supervised machine learning model that learns iteratively to recognize patterns and make predictions. By processing data layer by layer, FFNNs excel in tasks ranging from classification to regression, making them a fundamental tool in AI and deep learning. In this repository, We try to build our own FFNN model from scratch. Through this implementation, we aim to gain a deeper understanding of FFNNs by constructing and training the model step by step, without relying on pre-built libraries.

## Table of Contents

1. [Requirements](#requirements)
2. [Set Up Virtual Environment](#setup-virtual-environment)
3. [How to Run](#how-to-run)
4. [Bonus](#bonus)
5. [Acknowledgements](#acknowledgements)

## Requirements

1. `Python 3.12.7` or any stable version
2. Python Dependency listed on `requirements.txt`

## Setup Virtual Environment

1. install `virtualenv`
2. ```sh
   virtualenv venv
   ```
3. on Mac:
   ```sh
   source venv/bin/activate
   ```
   on Windows:
   ```sh
   venv/scripts/activate
   ```

## How To Run

1. Setup virtual environment
2. ```sh
   pip install -r src/requirements.txt
   ```
3. Setup kernel on `src/main.ipynb`
4. Adjust model configuration and data
5. run all!

## Bonus

1. Xavier Weight Initiation

## Acknowledgements

| Features                                                   | PIC                          |
| ---------------------------------------------------------- | ---------------------------- |
| Forward Propagation                                        | 13522164                     |
| Backward Propagation                                       | 13522134                     |
| Loss Function                                              | 13522157, 13522134           |
| Activation Function                                        | 13522164, 13522134           |
| Weight Initiation                                          | 13522164                     |
| Fit, Predict, Save, dan Load                               | 13522164, 13522134, 13522157 |
| Visualisasi Graf, Distribusi Bobot, dan Distribusi Gradien | 13522157                     |

13522134 - Sabrina Maharani  
13522157 - Muhammad Davis Adhipramana  
13522164 - Valentino Chryslie Triadi  