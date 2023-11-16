import sys
import pytest
from src.train import parse_parameters


def test_parse_arguments():
    # Test with valid arguments
    sys.argv = ["script.py", "data.csv", "10", "5"]
    assert parse_parameters() == ("data.csv", 10, 5)

    # Test with missing arguments
    sys.argv = ["script.py", "data.csv"]
    with pytest.raises(IndexError):
        parse_parameters()

    # Test with invalid arguments
    sys.argv = ["script.py", "data.csv", "invalid", "5"]
    with pytest.raises(ValueError):
        parse_parameters()
