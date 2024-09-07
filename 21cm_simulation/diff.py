#! /usr/bin/env python3
import sys

# Get first two command-line arguments
if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("Usage: python diff.py <file1> <file2> [<tolerance>]")
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]
if file1 == file2:
    print("Error: The two files are the same.")
    sys.exit(1)

tolerance = 1e-6
if len(sys.argv) == 4:
    tolerance = float(sys.argv[3])

with open(file1, 'r') as f1, open(file2, 'r') as f2:
    data1 = f1.readlines()
    data2 = f2.readlines()

# Check if the number of lines are the same
if len(data1) != len(data2):
    print("Error: The number of lines in the two files are different.")
    sys.exit(1)

# Compare each line
for line1, line2 in zip(data1, data2):
    if line1.strip() == line2.strip():
        continue
    else:
        for i, (val1, val2) in enumerate(zip(line1.split(), line2.split())):
            if abs(float(val1) - float(val2)) > tolerance:
                print(f"Difference found at line {data1.index(line1) + 1}:")
                print(f"File 1: {line1.strip()}")
                print(f"File 2: {line2.strip()}")
                exit(1)

print("No differences found.")
