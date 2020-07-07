import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from ArgparseTemplate import run_file, save_file, data_file

def main():
    if run_file == "House":
        execfile('HousePredictions.py')
    elif run_file == "Titanic":
        execfile('TitanicPredictions.py')
    else:
        print("Please enter a valid argument.")


main()
