import csv
from typing import IO, List


def make_batch_input_file(urls: List[str], file: IO[str]):
    writer = csv.DictWriter(file, fieldnames=["id", "url"])
    writer.writeheader()
    for i, url in enumerate(urls):
        writer.writerow({"id": i, "url": url})
