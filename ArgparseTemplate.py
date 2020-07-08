import argparse

#Trying to put this in a function
#Haven't figure out how to do that while still being able to call the variables run_file, data_file and save_file in Decision.py

parser = argparse.ArgumentParser(description='Enter House or Titanic to run the respective files')
parser.add_argument('python_file')
parser.add_argument('input_file')
parser.add_argument('output_file')
args = parser.parse_args()
run_file = args.python_file
data_file = args.input_file
save_file = args.output_file

