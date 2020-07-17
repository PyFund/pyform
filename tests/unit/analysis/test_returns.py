from pyform.analysis import table_calendar_return
from pyform import ReturnSeries

returns = ReturnSeries.read_csv("tests/unit/data/twitter_returns.csv")


def test_calendar_return():

    calendar_return = table_calendar_return(returns)
    assert (
        calendar_return.columns
        == [
            "Year",
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            "Total",
        ]
    ).all()

    assert calendar_return.iloc[0, 0] == 2013
