import torch
import argparse
import numpy as np
# gausian psf 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="", required=True)

    args = parser.parse_args()
    try:
        np.load(args.file)
        print("File is valid")
    except: 
        print(np.load("../" + args.file))
        
        print("File is invalid")

if __name__ == "__main__":
    main()