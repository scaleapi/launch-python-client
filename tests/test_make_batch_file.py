import csv
import tempfile

from launch.make_batch_file import make_batch_input_file


def test_make_batch_file():
    with tempfile.NamedTemporaryFile("w") as f:
        urls = ["one_url.count", "two_urls.count", "three_urls.count"]
        make_batch_input_file(urls, f)

        with open(f.name, "r") as f_read:
            reader = csv.DictReader(f_read)
            rows = [row for row in reader]
            print(rows)
            for i, expected_row, actual_row in zip(*enumerate(urls), rows):
                assert i == actual_row["id"]
                assert expected_row == actual_row["url"]
