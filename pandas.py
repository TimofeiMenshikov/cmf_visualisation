import csv
import json

import numpy as np


class Series:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __eq__(self, other):
        return self.values == other

    def __ne__(self, other):
        return self.values != other

    def __gt__(self, other):
        return self.values > other

    def __ge__(self, other):
        return self.values >= other

    def __lt__(self, other):
        return self.values < other

    def __le__(self, other):
        return self.values <= other

    def to_numpy(self, dtype=None):
        return self.values.astype(dtype) if dtype is not None else self.values

    def fillna(self, value):
        filled = []
        for item in self.values:
            if item != item:
                filled.append(value)
            else:
                filled.append(item)
        return Series(filled)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class DataFrame:
    def __init__(self, data):
        self._data = data

    @property
    def columns(self):
        if isinstance(self._data, dict):
            return list(self._data.keys())
        if self._data:
            keys = []
            for key in self._data[0].keys():
                if key != "__index__":
                    keys.append(key)
            return keys
        return []

    @property
    def index(self):
        if hasattr(self, "_index"):
            return Series(self._index)
        if isinstance(self._data, dict):
            return Series(list(self._data.keys()))
        return Series(list(range(len(self._data))))

    @property
    def T(self):
        if not isinstance(self._data, dict):
            return self
        rows = []
        for index, values in self._data.items():
            row = {"__index__": index}
            for column, value in enumerate(values):
                row[column] = value
            rows.append(row)
        frame = DataFrame(rows)
        frame._index = list(self._data.keys())
        return frame

    @property
    def values(self):
        if isinstance(self._data, dict):
            return np.asarray(list(self._data.values()))
        return np.asarray([[row[column] for column in self.columns] for row in self._data])

    def __getitem__(self, key):
        if isinstance(key, (str, int)):
            if isinstance(self._data, dict):
                return Series([value[key] if isinstance(value, dict) else value for value in self._data.values()])
            return Series([row[key] for row in self._data])

        mask = np.asarray(key)
        return DataFrame([row for row, keep in zip(self._data, mask) if keep])

    def __setitem__(self, key, value):
        values = value.values if isinstance(value, Series) else value
        if np.isscalar(values):
            values = [values] * len(self._data)
        for row, item in zip(self._data, values):
            row[key] = item

    def dropna(self, axis=0, how=None):
        if axis != 1:
            return self
        columns = self.columns
        kept_columns = []
        for column in columns:
            values = [row.get(column) for row in self._data]
            if how == "all" and all(value != value for value in values):
                continue
            kept_columns.append(column)
        return DataFrame([{column: row.get(column) for column in kept_columns} for row in self._data])

    def rename(self, columns=None, inplace=False):
        columns = columns or {}
        renamed = []
        for row in self._data:
            renamed.append({columns.get(key, key): value for key, value in row.items()})
        if inplace:
            self._data = renamed
            return None
        return DataFrame(renamed)

    def groupby(self, column):
        return GroupBy(self, column)

    def to_dict(self, orient="dict"):
        if orient == "records" and not isinstance(self._data, dict):
            return list(self._data)
        return self._data


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return DataFrame(json.load(file))


def read_csv(path, delimiter=",", decimal="."):
    rows = []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = np.nan
                    continue
                normalized = value.replace(decimal, ".") if decimal != "." else value
                try:
                    parsed[key] = float(normalized)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return DataFrame(rows)


class GroupBy:
    def __init__(self, dataframe, column):
        self.dataframe = dataframe
        self.column = column

    def max(self):
        groups = {}
        for row in self.dataframe._data:
            groups.setdefault(row[self.column], []).append(row)

        rows = []
        for key, group_rows in groups.items():
            max_row = {self.column: key}
            for column in self.dataframe.columns:
                values = [row[column] for row in group_rows]
                try:
                    max_row[column] = max(values)
                except TypeError:
                    max_row[column] = values[0]
            rows.append(max_row)

        result = DataFrame(rows)
        result._index = list(groups.keys())
        return result
