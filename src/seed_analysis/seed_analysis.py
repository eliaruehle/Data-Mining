import fnmatch
import os
import pprint


class Seed_Analysis:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.csv_files = self.find_csv_files()

    def find_csv_files(self):
        csv_files = {}
        metrics = [
            "accuracy",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1_score",
        ]
        # ! IMPORTANT: The "strategies" is the name of the folder where all Active Learning Strategies reside, must be adjusted if names dont match
        strategies_path = os.path.join(self.file_path, "strategies")

        for root, dirs, files in os.walk(strategies_path):
            for metric in metrics:
                for filename in fnmatch.filter(files, f"{metric}.csv.xz"):
                    if metric not in csv_files:
                        csv_files[metric] = []
                    csv_files[metric].append(os.path.join(root, filename))

        return csv_files

    def pretty_print_csv_files(self):
        pprint.pprint(self.csv_files)


seed = Seed_Analysis("kp_test")
seed.pretty_print_csv_files()
