import subprocess
import argparse

def main():
    file_name = raw_input("Would you like to run HousePredictions or TitanicPredictions?")
    if file_name == "HousePredictions":
        input_house = raw_input("Please enter input csv file:")
        output_house = raw_input("Please enter output file:")
        import HousePredictions
        args.input_file = input_house
        args.output_file = output_house
    elif file_name == "TitanicPredictions":
        input_titanic = raw_input("Please enter input csv file:")
        output_titanic = raw_input("Please enter output file:")
        import TitanicPredictions
    else:
        print("Please give a valid argument.")


main()
