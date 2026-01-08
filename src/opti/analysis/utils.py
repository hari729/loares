import pandas as pd
import json

def dict_to_csv(dictionary, filepath, filename):
    df = pd.DataFrame.from_dict(dictionary)
    df.to_csv(f"{filepath}/{filename}.csv", index=False)

def dict_to_json(dictionary, filepath, filename):
    path = f"{filepath}/{filename}.json"
    with open(path, "w") as f:
        json.dump(dictionary, f, indent=2)

def modify_master_list(master_list, filepath):
    mf = pd.DataFrame(master_list)
    master_list_path = filepath
    if master_list_path.exists():
        existing = pd.read_csv(master_list_path)
        combined = pd.concat([existing, mf], ignore_index=True)
        combined.drop_duplicates(subset=["algorithm", "psize", "max_evals", "seed"],
                                keep="last", inplace=True)
        combined.to_csv(master_list_path, index=False)
    else:
        mf.to_csv(master_list_path, mode='w', header=True, index = False)
