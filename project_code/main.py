from datasets.loader import Loader
from side_handler.errors import NoSuchPathOrCSV


def main():
    data = Loader("kp_test")

    print(data.get_single_dataframe("ALIPY_RANDOM", "Iris", "accuracy"))


if __name__ == "__main__":
    main()
