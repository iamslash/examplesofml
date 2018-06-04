# Abstract

If you train with historical datas of the stock then you can predict its closed price.

# Prerequisites

* python3
* tensorflow
* pandas-datareader
* fix_yahoo_finance

# Install

```
pip install -r requirements.txt
```

# Usage

## Train

```bash
t.py 001040.KS
```

Check the downloaded data `a.dat`. If there are datas to download then just download and train them.

## Predict

```bash
p.py 001040.KS
```

Predict the closed price of the first argument stock code on the next day of the last day in `a.dat`.