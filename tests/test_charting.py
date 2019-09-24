from ..code.charting import chart_to_image, chart_to_arr
import pandas as pd

candles = pd.read_csv('tests/sample_candles.csv')

def test_chart_image():
    chart_to_image(candles, 'tests/test_image.png')

def test_chart_arr():
    arr = chart_to_arr(candles)
    assert arr.shape == (3, 224, 224)