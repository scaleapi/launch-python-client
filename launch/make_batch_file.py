import csv
import json
from typing import IO, Any, Dict, List


def make_batch_input_file(urls: List[str], file: IO[str]):
    writer = csv.DictWriter(file, fieldnames=["id", "url"])
    writer.writeheader()
    for i, url in enumerate(urls):
        writer.writerow({"id": i, "url": url})


def make_batch_input_dict_file(inputs: List[Dict[str, Any]], file: IO[str]):
    writer = csv.DictWriter(file, fieldnames=["id", "args"], quotechar="'")
    writer.writeheader()
    for i, args in enumerate(inputs):
        writer.writerow({"id": i, "args": json.dumps(args)})
