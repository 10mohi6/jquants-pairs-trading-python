# jquants-pairs-trading

[![PyPI](https://img.shields.io/pypi/v/jquants-pairs-trading)](https://pypi.org/project/jquants-pairs-trading/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/10mohi6/jquants-pairs-trading-python/graph/badge.svg?token=X8QKKFK6AL)](https://codecov.io/gh/10mohi6/jquants-pairs-trading-python)
[![Python package](https://github.com/10mohi6/jquants-pairs-trading-python/actions/workflows/python-package.yml/badge.svg)](https://github.com/10mohi6/jquants-pairs-trading-python/actions/workflows/python-package.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jquants-pairs-trading)](https://pypi.org/project/jquants-pairs-trading/)
[![Downloads](https://pepy.tech/badge/jquants-pairs-trading)](https://pepy.tech/project/jquants-pairs-trading)

jquants-pairs-trading is a python library for backtest with japanese stock pairs trading using kalman filter, J-Quants on Python 3.8 and above.


## Installation

    $ pip install jquants-pairs-trading

## Usage

### find pairs

```python
from jquants_pairs_trading import JquantsPairsTrading
import pprint

jpt = JquantsPairsTrading(
    mail_address="<your J-Quants mail address>",
    password="<your J-Quants password>",
)
pprint.pprint(jpt.find_pairs([3382, 4063, 4502]))
```

![pairs.png](https://raw.githubusercontent.com/10mohi6/jquants-pairs-trading-python/main/tests/pairs.png)

```python
[('3382', '4502')]
```

### backtest

```python
from jquants_pairs_trading import JquantsPairsTrading
import pprint

jpt = JquantsPairsTrading(
    mail_address="<your J-Quants mail address>",
    password="<your J-Quants password>",
)
pprint.pprint(jpt.backtest((3382, 4502)))
```

![performance.png](https://raw.githubusercontent.com/10mohi6/jquants-pairs-trading-python/main/tests/performance.png)

```python
{'cointegration': '0.016',
 'correlation': '0.814',
 'maximum_drawdown': '443.000',
 'profit_factor': '1.654',
 'riskreward_ratio': '1.081',
 'sharpe_ratio': '0.183',
 'total_profit': '2184.000',
 'total_trades': '86.000',
 'win_rate': '0.605'}
```

### latest signal

```python
from jquants_pairs_trading import JquantsPairsTrading
import pprint

jpt = JquantsPairsTrading(
    mail_address="<your J-Quants mail address>",
    password="<your J-Quants password>",
)
pprint.pprint(jpt.latest_signal((6954, 6981)))
```

```python
{'6954 buy': True,
 '6954 close': '4348.000',
 '6954 long': False,
 '6954 sell': False,
 '6954 short': False,
 '6981 buy': False,
 '6981 close': '2775.000',
 '6981 long': False,
 '6981 sell': True,
 '6981 short': False,
 'date': '2023-07-31'}
```

### advanced

```python
from jquants_pairs_trading import JquantsPairsTrading
import pprint

jpt = JquantsPairsTrading(
    mail_address="<your J-Quants mail address>",
    password="<your J-Quants password>",
    window=1,
    transition_covariance=0.01,
    pvalues=0.05,
    zscore=0.5,
)
pprint.pprint(jpt.find_pairs([3382, 4063, 4502]))
pprint.pprint(jpt.backtest((3382, 4502)))
pprint.pprint(jpt.latest_signal((6954, 6981)))
```

## Getting started

For help getting started with J-Quants, view our online [documentation](https://jpx-jquants.com/).
