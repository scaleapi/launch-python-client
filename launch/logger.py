import logging
import typing

import requests

logger = logging.getLogger(__name__)
logging.basicConfig()

if typing.TYPE_CHECKING:
    # Ignore the following code because `requests.packages` does not pass mypy
    pass
else:
    logging.getLogger(requests.packages.urllib3.__package__).setLevel(logging.ERROR)  # pylint: disable=no-member
