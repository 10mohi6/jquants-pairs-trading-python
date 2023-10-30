import os
from enum import IntEnum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

from pykalman import KalmanFilter
from statsmodels.tsa.stattools import coint

import jquantsapi
import datetime


class Col(IntEnum):
    S1 = 0
    S2 = 1
    ZSCORE = 2
    S1BUY = 3
    S1SELL = 4
    S1PROFIT = 5
    S1PERFORMANCE = 6
    S2BUY = 7
    S2SELL = 8
    S2PROFIT = 9
    S2PERFORMANCE = 10


class JquantsPairsTrading:
    def __init__(
        self,
        *,
        mail_address: str = "",
        password: str = "",
        outputs_dir_path: str = "outputs",
        data_dir_path: str = "data",
        window: int = 1,
        zscore: float = 0.5,
        pvalues: float = 0.05,
        transition_covariance: float = 0.01,
    ) -> None:
        self.outputs_dir_path = outputs_dir_path
        self.data_dir_path = data_dir_path
        self.window = window
        self.zscore = zscore
        self.pvalues = pvalues
        self.transition_covariance = transition_covariance
        os.makedirs(self.outputs_dir_path, exist_ok=True)
        os.makedirs(self.data_dir_path, exist_ok=True)
        self.cli = jquantsapi.Client(mail_address=mail_address, password=password)

    def _get_prices_daily_quotes(self, ticker) -> pd.DataFrame:
        csv = "{}/{}-{}.csv".format(
            self.data_dir_path, ticker, str(datetime.date.today())
        )
        if os.path.isfile(csv):
            df = pd.read_csv(csv, parse_dates=True, index_col=0)
        else:
            df = (
                self.cli.get_prices_daily_quotes(ticker)[
                    [
                        "Date",
                        "AdjustmentClose",
                    ]
                ]
                .rename(
                    columns={
                        "AdjustmentClose": ticker,
                    }
                )
                .set_index("Date")
                .fillna(method="bfill")
            )
            df.to_csv(csv)
        return df

    def _is_exit(self, df: pd.DataFrame, i: int) -> bool:
        return abs(df.iat[i, Col.ZSCORE]) < self.zscore or (
            df.iat[i - 1, Col.ZSCORE] > self.zscore
            and df.iat[i, Col.ZSCORE] < -self.zscore
            or (
                df.iat[i - 1, Col.ZSCORE] < -self.zscore
                and df.iat[i, Col.ZSCORE] > self.zscore
            )
        )

    def latest_signal(self, pair: tuple) -> dict:
        s1 = str(pair[0])
        s2 = str(pair[1])
        df = self._get_prices_daily_quotes(s1)
        df = df.join(self._get_prices_daily_quotes(s2))
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=self.transition_covariance,
        )
        state_means, state_cov = kf.filter(df[s1] / df[s2])
        state_means, state_std = state_means.squeeze(), np.std(state_cov.squeeze())
        ma = (df[s1] / df[s2]).rolling(window=self.window, center=False).mean()
        df["zscore"] = (ma - state_means) / state_std
        r = {}
        r["date"] = df.index[-1].strftime("%Y-%m-%d")
        r["{} close".format(s1)] = "{:.3f}".format(df.iat[-1, Col.S1])
        r["{} close".format(s2)] = "{:.3f}".format(df.iat[-1, Col.S2])
        # r["zscore"] = "{:.3f}".format(df.iat[-1, Col.ZSCORE])
        r["{} buy".format(s1)] = df.iat[-1, Col.ZSCORE] < -1
        r["{} long".format(s1)] = self._is_exit(df, -1)
        r["{} sell".format(s1)] = df.iat[-1, Col.ZSCORE] > 1
        r["{} short".format(s1)] = self._is_exit(df, -1)
        r["{} buy".format(s2)] = df.iat[-1, Col.ZSCORE] > 1
        r["{} long".format(s2)] = self._is_exit(df, -1)
        r["{} sell".format(s2)] = df.iat[-1, Col.ZSCORE] < -1
        r["{} short".format(s2)] = self._is_exit(df, -1)
        return r

    def find_pairs(self, tickers: list) -> list:
        df = self._get_prices_daily_quotes(str(tickers[0]))
        for i in tickers[1:]:
            df = df.join(self._get_prices_daily_quotes(str(i)))
        _, pvalues, pairs = self._find_cointegrated_pairs(df)
        plt.figure(figsize=(15, 7))
        seaborn.heatmap(
            pvalues,
            xticklabels=tickers,
            yticklabels=tickers,
            cmap="RdYlGn_r",
            mask=(pvalues >= self.pvalues),
        )
        plt.savefig("{}/pairs.png".format(self.outputs_dir_path))
        plt.clf()
        plt.close()
        return pairs

    def _find_cointegrated_pairs(self, data: pd.DataFrame) -> Any:
        n = data.shape[1]
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        keys = data.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                result = coint(S1, S2)
                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05:
                    pairs.append((keys[i], keys[j]))
        return score_matrix, pvalue_matrix, pairs

    def backtest(
        self,
        pair: tuple,
    ) -> dict:
        s1 = str(pair[0])
        s2 = str(pair[1])
        df = self._get_prices_daily_quotes(s1)
        df = df.join(self._get_prices_daily_quotes(s2))
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=self.transition_covariance,
        )
        state_means, state_cov = kf.filter(df[s1] / df[s2])
        state_means, state_std = state_means.squeeze(), np.std(state_cov.squeeze())
        ma = (df[s1] / df[s2]).rolling(window=self.window, center=False).mean()
        df["zscore"] = (ma - state_means) / state_std
        df["s1Buy"] = df["s1Sell"] = df["s1Profit"] = df["s1Performance"] = np.nan
        df["s2buy"] = df["s2Sell"] = df["s2Profit"] = df["s2Performance"] = np.nan
        s1Profit = [np.nan, np.nan]
        s2Profit = [np.nan, np.nan]
        s1Performance = s2Performance = 0.0
        flag = 0
        for i in range(len(df)):
            if self._is_exit(df, i):
                if flag == 1:
                    if not np.isnan(s1Profit[0]):
                        df.iat[i, Col.S1SELL] = df.iat[i, Col.S1]
                        df.iat[i, Col.S1PROFIT] = df.iat[i, Col.S1SELL] - s1Profit[0]
                    elif not np.isnan(s1Profit[1]):
                        df.iat[i, Col.S1BUY] = df.iat[i, Col.S1]
                        df.iat[i, Col.S1PROFIT] = s1Profit[1] - df.iat[i, Col.S1BUY]
                    s1Profit = [np.nan, np.nan]
                    s1Performance += df.iat[i, Col.S1PROFIT]
                    if not np.isnan(s2Profit[0]):
                        df.iat[i, Col.S2SELL] = df.iat[i, Col.S2]
                        df.iat[i, Col.S2PROFIT] = df.iat[i, Col.S2SELL] - s2Profit[0]
                    elif not np.isnan(s2Profit[1]):
                        df.iat[i, Col.S2BUY] = df.iat[i, Col.S2]
                        df.iat[i, Col.S2PROFIT] = s2Profit[1] - df.iat[i, Col.S2BUY]
                    s2Profit = [np.nan, np.nan]
                    s2Performance += df.iat[i, Col.S2PROFIT]
                flag = 0
            elif df.iat[i, Col.ZSCORE] > 1:
                if flag == 0:
                    df.iat[i, Col.S1SELL] = df.iat[i, Col.S1]
                    s1Profit = [np.nan, df.iat[i, Col.S1SELL]]
                    df.iat[i, Col.S2BUY] = df.iat[i, Col.S2]
                    s2Profit = [df.iat[i, Col.S2BUY], np.nan]
                    flag = 1
            elif df.iat[i, Col.ZSCORE] < -1:
                if flag == 0:
                    df.iat[i, Col.S1BUY] = df.iat[i, Col.S1]
                    s1Profit = [df.iat[i, Col.S1BUY], np.nan]
                    df.iat[i, Col.S2SELL] = df.iat[i, Col.S2]
                    s2Profit = [np.nan, df.iat[i, Col.S2SELL]]
                    flag = 1

            df.iat[i, Col.S1PERFORMANCE] = s1Performance
            df.iat[i, Col.S2PERFORMANCE] = s2Performance
        fig = plt.figure(figsize=((6.4 * 2.5), (4.8 * 2.5)))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.plot(df.index, df["s1Performance"].values, alpha=0.5)
        ax1.plot(df.index, df["s2Performance"].values, alpha=0.5)
        ax1.plot(df.index, df["s1Performance"].values + df["s2Performance"].values)
        ax1.grid()
        ax1.legend(
            [
                "{} profit".format(s1),
                "{} profit".format(s2),
                "{} + {} profit".format(s1, s2),
            ]
        )
        ax2.plot(df.index, (df[s1] - df[s1][0]).values, alpha=0.5)
        ax2.plot(df.index, (df[s2] - df[s2][0]).values, alpha=0.5)
        ax2.plot(df.index, (df[s1] - df[s1][0]).values + (df[s2] - df[s2][0]).values)
        ax2.grid()
        ax2.legend(
            [
                "{} close".format(s1),
                "{} close".format(s2),
                "{} + {} close".format(s1, s2),
            ]
        )
        fig.tight_layout()
        plt.savefig("{}/performance.png".format(self.outputs_dir_path))
        plt.clf()
        plt.close()
        df.to_csv("{}/performance.csv".format(self.outputs_dir_path))
        _, pvalue, _ = coint(df[s1], df[s2])
        score = df[s1].corr(df[s2])
        win_num = (df["s1Profit"] > 0).sum() + (df["s2Profit"] > 0).sum()
        loss_num = (df["s1Profit"] <= 0).sum() + (df["s2Profit"] <= 0).sum()
        total_trades = win_num + loss_num
        win = (
            df["s1Profit"].where(df["s1Profit"] > 0, 0).sum()
            + df["s2Profit"].where(df["s2Profit"] > 0, 0).sum()
        )
        loss = (
            df["s1Profit"].where(df["s1Profit"] <= 0, 0).sum()
            + df["s2Profit"].where(df["s2Profit"] <= 0, 0).sum()
        )
        total_profit = df["s1Profit"].sum() + df["s2Profit"].sum()
        win_rate = win_num / total_trades
        profit_factor = win / abs(loss)
        average_win = win / win_num
        average_loss = abs(loss) / loss_num
        riskreward_ratio = average_win / average_loss
        mdd = (
            np.maximum.accumulate(df["s1Performance"] + df["s2Performance"])
            - (df["s1Performance"] + df["s2Performance"])
        ).max()
        sharpe_ratio = (df["s1Profit"].mean() + df["s2Profit"].mean()) / (
            df["s1Profit"].std() + df["s2Profit"].std()
        )
        r = {}
        r["cointegration"] = "{:.3f}".format(pvalue)
        r["correlation"] = "{:.3f}".format(score)
        r["total_profit"] = "{:.3f}".format(total_profit)
        r["total_trades"] = "{:.3f}".format(total_trades)
        r["win_rate"] = "{:.3f}".format(win_rate)
        r["profit_factor"] = "{:.3f}".format(profit_factor)
        r["riskreward_ratio"] = "{:.3f}".format(riskreward_ratio)
        r["sharpe_ratio"] = "{:.3f}".format(sharpe_ratio)
        r["maximum_drawdown"] = "{:.3f}".format(mdd)
        return r
