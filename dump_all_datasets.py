import sys
from data_loader import get_dataset

DATA_NAMES = [
    "abalone",
    "airline",
    "airline-one-hot",
    "epsilon",
    "higgs",
    "letters",
    "msrank",
    "msrank-classification"
]

if __name__ == "__main__":
    out_dir = sys.argv[1]
    print('out_dir: ' + str(out_dir))

    for dataset_name in DATA_NAMES:
        print('Processing ' + dataset_name)
        get_dataset(dataset_name, out_dir)
