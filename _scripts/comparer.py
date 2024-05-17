import re
import argparse
import numpy as np


def read_complex_numbers(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'\[(\d+)\]\s*(\()?(.*)\s*\+\s*(.*)(j|i).*', line)
            if match:
                groups = match.groups()
                real_part = float(groups[2])
                imag_str = groups[3].replace('j', '').replace('i', '')
                imag_part = float(imag_str)
                numbers.append(complex(real_part, imag_part))

    return numbers


def read_numbers(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'\[(\d+)\]\s*(.*)\s*', line)
            if match:
                groups = match.groups()
                numbers.append(float(groups[1]))
    return numbers


def compare_files(file1, file2, print_errors=False):
    numbers1 = read_complex_numbers(file1)
    numbers2 = read_complex_numbers(file2)

    max_abs_error = 0
    max_relative_error = 0
    printed_errors = 0
    for i in range(min(len(numbers1), len(numbers2))):
        # print(f"[{i}] Comparing {numbers1[i]} and {numbers2[i]}")
        abs_error = abs(numbers1[i] - numbers2[i])
        if abs_error > max_abs_error:
            max_abs_error = abs_error
            max_relative_error = abs_error / max(abs(numbers1[i]),
                                                 abs(numbers2[i]))
        if abs_error > 51 and print_errors and printed_errors < 10:
            print(f"Error at index {i}: {abs_error}")
            printed_errors += 1

    numbers1 = np.array(numbers1)
    numbers2 = np.array(numbers2)
    relative_error_to_max = max_abs_error / max(abs(numbers1) + abs(numbers2))
    relative_error_to_mean = max_abs_error / np.mean(
        abs(numbers1) + abs(numbers2))

    return max_abs_error, max_relative_error, relative_error_to_max, relative_error_to_mean


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Comparer')
    parser.add_argument('-f',
                        '--files',
                        nargs=2,
                        help='Files to compare',
                        required=True)
    parser.add_argument('-p',
                        '--print',
                        help='Print errors',
                        action='store_true')

    args = parser.parse_args()

    max_abs_error, max_relative_error, relative_error_to_max, relative_error_to_mean = compare_files(
        args.files[0], args.files[1], args.print)
    print("Max absolute error without:", max_abs_error)
    print("Relative error:", max_relative_error)
    print("Relative error to max:", relative_error_to_max)
    print("Relative error to mean:", relative_error_to_mean)
