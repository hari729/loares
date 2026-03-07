import pandas as pd
import json


def dict_to_csv(dictionary, filepath, filename):
    df = pd.DataFrame.from_dict(dictionary)
    df.to_csv(f"{filepath}/{filename}.csv", index=False)


def dict_to_json(dictionary, filepath, filename):
    path = f"{filepath}/{filename}.json"
    with open(path, "w") as f:
        json.dump(dictionary, f, indent=2)
