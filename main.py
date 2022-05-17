from data.load_data import get_data
from model.model import run_model
import sys


def main():
    arg = sys.argv[1]
    if arg == '--download-data':
        get_data()
    if arg == '--run-model':
        run_model()
    if arg == '--run-all':
        get_data()
        run_model()


if __name__ == '__main__':
    main()
