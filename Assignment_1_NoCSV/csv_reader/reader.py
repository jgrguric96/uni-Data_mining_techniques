import csv


class reader:
    @staticmethod
    def get_csv(path: str):
        rows = []
        with open(path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                rows.append(row)
        return rows
