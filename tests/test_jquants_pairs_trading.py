import pandas as pd
import pytest

from jquants_pairs_trading import JquantsPairsTrading


@pytest.fixture(scope="module", autouse=True)
def scope_module():
    yield JquantsPairsTrading(
        mail_address="dummy@dummy",
        password="dummy",
        outputs_dir_path="tests",
        data_dir_path="tests",
    )


@pytest.fixture(scope="function", autouse=True)
def jpt(scope_module, mocker):
    mocker.patch(
        "jquants_pairs_trading.JquantsPairsTrading._get_prices_daily_quotes",
        side_effect=lambda x: pd.read_csv(
            "tests/3382-2023-10-23.csv", index_col=0, parse_dates=True
        )
        if x == "3382"
        else (
            pd.read_csv("tests/4063-2023-10-23.csv", index_col=0, parse_dates=True)
            if x == "4063"
            else pd.read_csv("tests/4502-2023-10-23.csv", index_col=0, parse_dates=True)
        ),
    )
    yield scope_module


# @pytest.mark.skip
def test_find_pairs(jpt):
    jpt.find_pairs([3382, 4063, 4502])


# @pytest.mark.skip
def test_backtest(jpt):
    jpt.backtest((3382, 4502))


# @pytest.mark.skip
def test_latest_signal(jpt):
    jpt.latest_signal((3382, 4502))
