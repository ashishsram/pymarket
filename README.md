# pymarket
-------------
finance portfolio visualizer
-------------
Portfolio displays how your stocks have performed over time
Distribution based on geography and sector

-------------
Uses libraries:
-------------
forex_python.converter
yfinance

-------------
Inputs:
-------------
Set following variables based on your data in market_watch.py:
invest_start_date = get_date('2019-01-01')
config_path = '/path/to/csv/file/with/all/your/stock/purchase/dates/and/quantity/'

-------------
Files:
-------------
market_watch.py: This is the main file that generates the portfolio dashboard
asset_data.py: Parse input csv file containing list of stocks purchased, quantity, etc
real_estate_ticker.py: Not yet ready
ut_ticker.py: Unit trust investment tracker, for now just reads data from a csv file
pymarket_utils.py: Utility functions, currency convertor, etc

-------------
References
-------------
#finance analysis in python/pandas/yfinance : https://render.githubusercontent.com/view/ipynb?commit=bd317ee6281f371b0a4a4bf5ccd42ac91566da56&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6461746163616d702f6461746163616d702d636f6d6d756e6974792d7475746f7269616c732f626433313765653632383166333731623061346134626635636364343261633931353636646135362f507974686f6e25323046696e616e63652532305475746f7269616c253230466f72253230426567696e6e6572732f507974686f6e253230466f7225323046696e616e6365253230426567696e6e6572732532305475746f7269616c2e6970796e62&nwo=datacamp%2Fdatacamp-community-tutorials&path=Python+Finance+Tutorial+For+Beginners%2FPython+For+Finance+Beginners+Tutorial.ipynb&repository_id=78221549&repository_type=Repository#basics
#pandas plots all: https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
#dataframe indexing and selecting : https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
#https://pypi.org/project/forex-python/
