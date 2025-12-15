import os
import pandas as pd
from src.utils.text import basic_clean

def test_basic_clean():
    assert basic_clean("  Hello!!  ") == "hello!!"
    assert basic_clean(None) == ""
