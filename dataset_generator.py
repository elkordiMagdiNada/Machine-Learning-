import random

def build_random():
    input_filename = 'Dbig.txt'
    output_filename = 'D8192.txt'
    subset_size = 8192
    with open(input_filename, 'r') as input_file:
        lines = input_file.readlines()
        subset = random.sample(lines, subset_size)

    with open(output_filename, 'w') as output_file:
        output_file.writelines(subset)

def build_first():
    input_filename = 'D8192.txt'
    output_filename = 'D32.txt'
    k = 32  # set the number of lines to read

    # Read the first k lines of the input file
    with open(input_filename, 'r') as input_file:
        lines = [next(input_file) for x in range(k)]

    # Write the selected lines to the output file
    with open(output_filename, 'w') as output_file:
        output_file.writelines(lines)

build_first()