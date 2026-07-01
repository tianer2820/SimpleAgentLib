#!/usr/bin/env python3
import sys

def print_triangle(height):
    for i in range(1, height + 1):
        print(" " * (height - i) + "o" * (2 * i - 1))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 triangle.py <height>")
        sys.exit(1)
    
    try:
        height = int(sys.argv[1])
        if height < 1:
            print("Please enter a positive integer.")
        else:
            print_triangle(height)
    except ValueError:
        print("Please enter a valid integer.")
