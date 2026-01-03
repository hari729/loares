import pandas as pd

def dict_to_csv(dictionary, filepath, filename):
    df = pd.DataFrame.from_dict(dictionary)
    df.to_csv(f"{filepath}/{filename}.csv", index=False)
