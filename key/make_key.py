import argparse
import random

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=int, required=True, help="size of key")
    parser.add_argument("-o", type=str, required=True, help="output file")
    return parser

def main():
    parser = parsing()
    args = parser.parse_args()
    
    key_values = [random.choice(range(-128, 0)) if random.random() < 0.5 else random.choice(range(1, 128)) for _ in range(args.s)]
    
    with open(args.o, 'wb') as file:
        for value in key_values:
            file.write(value.to_bytes(1, byteorder='big', signed=True))

main()