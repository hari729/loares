import json
from pathlib import Path

def compare(list_of_tests,problem,algo,psize):
    results_path = Path(__file__).resolve().parent.parent/'results'
    for test in list_of_tests:
        with open(f"{results_path}/{test}/{problem}/{algo.upper()}/{psize}/settings.json") as f:
            data = json.load(f)

        print(data)

if __name__ == "__main__":

    compare(["json_test_20251016_145520"],"mou.zdt2","bwr","100")
