import calendar
import pandas as pd
from pyform import ReturnSeries


def table_calendar_return(
    return_series: ReturnSeries, use_month_abbr: bool = True
) -> pd.DataFrame:
    """Create calendar like monthly return table

    Args:
        return_series: A return series. Should be of minimum monthly frequency
        use_month_abbr: Whether to use 3 letter month abbreviations instead of numerical
          month. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with columns: Year, Jan, Feb, ..., Dec, Total
    """

    # create monthly and annual returns for output rows
    monthly = return_series.to_month()
    annual = return_series.to_year()

    # append necessary information
    monthly["Year"] = monthly.index.year
    monthly["Month"] = monthly.index.month
    monthly["Month"] = monthly["Month"]
    annual["Year"] = annual.index.year

    # Pivot monthly data
    monthly = monthly.pivot(index="Year", columns="Month", values=monthly.columns[0])

    # Rename annual column, and merge with monthly
    annual = annual.rename(columns={annual.columns[0]: "Total"})

    output = monthly.merge(annual, how="left", on="Year")

    if use_month_abbr:
        months = [*range(1, 13)]
        month_abbr_map = {month: calendar.month_abbr[month] for month in months}
        output = output.rename(columns=month_abbr_map)

    return output
