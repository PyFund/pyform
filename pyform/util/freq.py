import pandas as pd
from typing import Optional


def is_lower_freq(freq1: str, freq2: str) -> bool:
    """Tests freq1 has a lower or equal frequency than freq2.
    Lower frequencies cannot be converted to higher frequencies
    due to lower resolution.

    Args:
        freq1: frequency 1
        freq2: frequency 2

    Returns:
        bool: frequency 1 is lower than or euqal to frequency 2
    """

    freq = ["H", "D", "B", "W", "M", "Q", "Y"]
    freq = dict(zip(freq, [*range(0, len(freq))]))

    return freq[freq1] >= freq[freq2]


def infer_freq(series: pd.DataFrame, use: Optional[int] = 50) -> str:
    """Infer the frequency of the time series

    Args:
        series: a pandas DataFrame with datetime index
        use: number of data points to use from the head and tail of the time series
            index in order to determin frequency. This should be at least 10. Defaults
            to 50.

    Raises:
        ValueError: when multiple frequencies are detected
        ValueError: when no frequency can be detected

    Returns:
        str: frequency of the time series
    """

    freq = set()
    max_check = min(len(series.index) - 10, use)

    if max_check <= 10:
        inferred_freq = series.index.inferred_freq
        if inferred_freq is not None:
            freq.add(inferred_freq)
    else:
        # check head
        for i in range(0, max_check, 10):

            inferred_freq = series.index[i : (i + 10)].inferred_freq

            if inferred_freq is not None:

                freq.add(inferred_freq)

        # check from tail
        for i in range(0, -max_check, -10):

            inferred_freq = series.index[(i - 11) : (i - 1)].inferred_freq

            if inferred_freq is not None:

                freq.add(inferred_freq)

    if len(freq) == 0:
        raise ValueError("Cannot infer series frequency.")

    if len(freq) > 1:
        raise ValueError(f"Multiple series frequency detected: {freq}")

    return freq.pop()


def calc_timedelta_in_years(start, end) -> float:
    """Computes timedelta between start and end dates, in years
        To use this function, start and end dates should come
        from a series of daily frequency.

    Args:
        start: start date
        end: end date

    Returns:
        float: time delta in number of years
    """

    # datetime representation of number of days in 1 year
    one_year = pd.to_timedelta(365.25, unit="D")
    one_day = pd.to_timedelta(1, unit="D")

    # Compute the duration of the series in terms of number of years
    years = (end - start + one_day) / one_year

    return years


def calc_samples_per_year(num_samples: int, start, end) -> float:
    """Computes number of data points per year, given time range

    Args:
        num_samples: total number of samples
        start: start date of samples
        end: end date of samples

    Returns:
        float: average number of samples per year
    """

    years = calc_timedelta_in_years(start, end)

    # Get number of data points per year
    samples_per_year = num_samples / years

    return samples_per_year
