import csv
from io import StringIO

from launch.make_batch_file import make_batch_input_file


def test_make_batch_file():
    f = StringIO()
    urls = ["one_url.count", "two_urls.count", "three_urls.count"]
    make_batch_input_file(urls, f)
    f.seek(0)

    reader = csv.DictReader(f)
    rows = [row for row in reader]
    print(f.getvalue())
    print(rows)
    for tup in zip(enumerate(urls), rows):
        print(tup)
        (i, expected_row), actual_row = tup
        assert str(i) == actual_row["id"]
        assert expected_row == actual_row["url"]
