#!/usr/bin/env python
# coding: utf-8

# In[147]:


#All packages can be installed using: pip install <package-name>
#Additional pip install lxml
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import datetime
import os
from forex_python.converter import CurrencyRates
from abc import ABC, abstractmethod

#Convert all dependent notebooks to python scripts
#Important: Add any new dependencies here
get_ipython().system('jupyter nbconvert --to script asset_data.ipynb')
get_ipython().system('jupyter nbconvert --to script pymarket_utils.ipynb')
get_ipython().system('jupyter nbconvert --to script real_estate_ticker.ipynb')
get_ipython().system('jupyter nbconvert --to script ut_ticker.ipynb')

#My modules
import ut_ticker
from pymarket_utils import currency_conv, get_date
from asset_data import AssetData, MyAssetDataDict
#import real_estate_ticker


# In[148]:


#References
#finance analysis in python/pandas/yfinance : https://render.githubusercontent.com/view/ipynb?commit=bd317ee6281f371b0a4a4bf5ccd42ac91566da56&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6461746163616d702f6461746163616d702d636f6d6d756e6974792d7475746f7269616c732f626433313765653632383166333731623061346134626635636364343261633931353636646135362f507974686f6e25323046696e616e63652532305475746f7269616c253230466f72253230426567696e6e6572732f507974686f6e253230466f7225323046696e616e6365253230426567696e6e6572732532305475746f7269616c2e6970796e62&nwo=datacamp%2Fdatacamp-community-tutorials&path=Python+Finance+Tutorial+For+Beginners%2FPython+For+Finance+Beginners+Tutorial.ipynb&repository_id=78221549&repository_type=Repository#basics
#pandas plots all: https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
#dataframe indexing and selecting : https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
#https://pypi.org/project/forex-python/


# In[149]:


run_tests = False


# In[150]:


my_stock_desc = [
    {'symbol' : '1211.HK' ,'sector':'EV'      ,'country':'China'        , 'acc':'investment'},
    {'symbol' : 'IXC'     ,'sector':'Oil'     ,'country':'United States', 'acc':'investment', 'geo_split':{'United States':0.51,'UK':0.14,'Canada':0.13,'EU':0.10}},
]

my_unit_trust_desc = [
    {'symbol':'SG9999000343U', 'name':'Schroder Singapore Trust A'      ,'currency':'SGD','sector':'All',
     'country':'Singapore', 'acc':'retirement', 'geo_split':{'Singapore':1.0}},
]

invest_start_date = get_date('2019-01-01')

config_path = '/mnt/c/ubuntu/my_assets/config/'

ut = ut_ticker.UtTicker(config_path + 'unit_trust')
my_asset_dict = MyAssetDataDict(config_path)
#rt = real_estate_ticker.Re_Ticker('/home/ashish/pymarket/notebooks/config/real_estate')


# In[151]:


#Test currency_conv
if run_tests:
    value = np.array([100.0, 50.0])
    u_to_s = currency_conv(value, 'USD', 'SGD')
    print("{0} USD = {1} SGD".format(value, u_to_s))
    s_to_u = currency_conv(u_to_s, 'SGD', 'USD')
    assert(abs(s_to_u - value) < value * 0.1).all()


# In[152]:


class Asset(ABC):
    def __init__(self):
        pass
    
    def _set_business(self, manual):
        self.sector = manual['sector']
        self.country = manual['country']
        self.account_type = manual['acc']
        self.geography = {}
        if 'geo_split' in manual:
            geo = manual['geo_split']
            tot = np.sum(list(geo.values()))
            assert tot <= 1.0, 'Invalid geo_split. Sum should be <= 1.0'
            if tot<1.0:
                self.geography['Others'] = 1.0 - tot
            for key, val in geo.items():
                self.geography[key] = val
        else:
            self.geography = {self.country : 1.0}
        
    def business_summary(self):
        return {
        'Name'     : self.name,
        'Symbol'   : self.symbol,
        'Sector'   : self.sector,
        'Geography': self.geography,
        'AccType'  : self.account_type
        }
        
    @abstractmethod
    def current_value(self):
        pass
    
    def _asset_trace(self, all_val, start_date, end_date, smooth=-1):
        """
        Trace must have range [start_date, end_date]
        """
        mask = (all_val.index >= pd.to_datetime(start_date))
        all_val = all_val.loc[mask]
        all_val = all_val[['Close']]
        
        idx = pd.date_range(start_date, end_date)
        all_val = all_val.reindex(idx) #missing date added (before stock exists, weekends, no data), etc
        all_val = all_val.fillna(method='ffill') #fwd fill missing for values in between
        all_val = all_val.fillna(0) #fill 0s for values before stock exists
        
        #all_val.rename(columns={'Close' : self.name}, inplace=True)
        
        if(smooth>0):
            all_val = all_val.rolling(smooth).mean()
            
        return all_val


# In[153]:


class Stock(Asset):
    def __init__(self, symbol, manual):
        self.ticker = yf.Ticker(symbol)
        assert self.ticker.history(period='1d')['Close'].size > 0, "Invalid symbol " + symbol
        self.manual = manual
        self.set_business()
    
    def set_business(self):
        self.name              = self.ticker.info['shortName']
        self.sector_market     = self.ticker.info['sector'] if 'sector'in self.ticker.info else None
        self.currency          = self.ticker.info['currency']
        self.symbol            = self.ticker.info['symbol']
        self.country_market    = self.ticker.info['country'] if 'country' in self.ticker.info else None
        super()._set_business(self.manual)
        
    def current_value(self):
        return self.ticker.history('1d')['Close'][0]
    
    def asset_trace(self, start_date, end_date, smooth=-1):
        all_val = self.ticker.history('max')
        return super()._asset_trace(all_val, start_date, end_date, smooth)


# In[154]:


#Test Stock
if run_tests:
    test_stock = Stock(my_stock_desc[0]['symbol'], my_stock_desc[0])
    print(test_stock.business_summary())
    print(test_stock.current_value())
    print(test_stock.asset_trace(get_date('1980-04-20'), pd.to_datetime('today')))


# In[155]:


class UnitTrust(Asset):
    def __init__(self, symbol, manual):
        self.symbol = symbol
        self.manual = manual
        self.ticker = ut.Ticker(self.symbol)
        self.set_business()
    
    def set_business(self):
        self.name              = self.manual['name']
        self.currency          = self.manual['currency']
        super()._set_business(self.manual)

    def current_value(self):
        return self.ticker['Close'][-1]
    
    def asset_trace(self, start_date, end_date, smooth=-1):
        return super()._asset_trace(self.ticker, start_date, end_date, smooth)


# In[156]:


#Test UnitTrust
if run_tests:
    test_ut = UnitTrust(my_unit_trust_desc[0]['symbol'], my_unit_trust_desc[0])
    print(test_ut.business_summary())
    print(test_ut.current_value())
    print(test_ut.asset_trace(get_date('1980-04-20'), pd.to_datetime('today')))


# In[157]:


class MyAsset(ABC):
    def __init__(self, symbol):
        self.asset_data = my_asset_dict.asset(symbol)
        
    def current_asset_value(self, currency=None):
        cur_org = self.asset.current_value() * self.asset_data.total_quantity()
        if(currency==None):
            return cur_org
        else:
            return currency_conv(cur_org, self.asset.currency, currency)

    def invested_asset_value(self, currency=None):
        inv_org = self.asset_data.total_invested()
        if(currency==None):
            return inv_org
        else:
            return currency_conv(inv_org, self.asset.currency, currency)
        
    def asset_summary(self, currency):
        return {'stock'    : self.asset.name,
               'value'     : self.current_asset_value(currency),
               'invested'  : self.invested_asset_value(currency),
               'sector'    : self.asset.sector,
               'geography' : self.asset.geography,
               'acc'       : self.asset.account_type,
               '% change'  : 0.0}
        
    def asset_trace(self, start_date, currency=None):
        end_date = pd.to_datetime('today')
        close = self.asset.asset_trace(start_date, end_date)
        quantity = self.asset_data.trace_quantity(start_date, end_date)
        assert len(close) == len(quantity), "asset and asset_data traces have unequal entries!"
        df = close["Close"] * quantity["Quantity"]
        df = pd.DataFrame(df, columns=[self.asset.name])
        if(currency==None):
            return df
        else:
            return currency_conv(df, self.asset.currency, currency)
        
    def investment_trace(self, start_date, currency=None):
        """
        idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime('today'))
        values = pd.DataFrame({'Date' : idx, self.asset.name : np.zeros(len(idx))})
        values = values.set_index('Date')
        mask = (values.index >= pd.to_datetime(self.purchase_date))
        values.loc[mask] = self.purchase_price
        values = values * self.quantity
        """
        values = self.asset_data.trace_invested(start_date, pd.to_datetime('today'))
        if(currency==None):
            return values
        else:
            return currency_conv(values, self.asset.currency, currency)


# In[158]:


class MyStock(MyAsset):
    def __init__(self, manual):
        symbol = manual['symbol']
        self.asset = Stock(symbol, manual)
        super().__init__(symbol)


# In[159]:


#Test MyStock
if run_tests:
    test_my_stock = MyStock(my_stock_desc[0])
    print(test_my_stock.current_asset_value('SGD'))
    test_my_stock.asset_trace(invest_start_date, 'SGD').plot()
    test_my_stock.investment_trace(invest_start_date, 'SGD').plot()


# In[160]:


class MyUnitTrust(MyAsset):
    def __init__(self, manual):
        symbol = manual['symbol']
        self.asset = UnitTrust(symbol, manual)
        super().__init__(symbol)


# In[161]:


#Test MyUnitTrust
if run_tests:
    test_my_ut = MyUnitTrust(my_unit_trust_desc[1])
    print(test_my_ut.current_asset_value('SGD'))
    test_my_ut.asset_trace(invest_start_date, 'SGD').plot()
    test_my_ut.investment_trace(invest_start_date, 'SGD').plot()


# In[162]:


class Portfolio:
    def __init__(self, stock_list, ut_list, currency):
        self.stocks = list(map(lambda x : MyStock(x), stock_list))
        self.uts = list(map(lambda x : MyUnitTrust(x), ut_list))
        self.currency = currency
    
    def asset_distibution(self):
        cur_assets_stocks = map(lambda x : x.asset_summary(self.currency), self.stocks)
        cur_assets_ut = map(lambda x : x.asset_summary(self.currency), self.uts)
        cur_assets = list(cur_assets_stocks) + list(cur_assets_ut)
        df = pd.DataFrame(columns = list(cur_assets[0].keys()))
        for i, asset in enumerate(cur_assets):
            asset['% change'] = (asset['value'] - asset['invested']) / asset['invested'] * 100.0
            df.loc[i] = list(asset.values())
        return df
    
    def asset_trace(self):
        traces_st = list(map(lambda x : x.asset_trace(invest_start_date, self.currency), self.stocks))
        traces_ut = list(map(lambda x : x.asset_trace(invest_start_date, self.currency), self.uts))
        traces = traces_st + traces_ut
        traces = pd.concat(traces, axis=1, sort=False)
        return traces
    
    def investment_trace(self):
        traces_st = list(map(lambda x : x.investment_trace(invest_start_date, self.currency), self.stocks))
        traces_ut = list(map(lambda x : x.investment_trace(invest_start_date, self.currency), self.uts))
        traces = traces_st + traces_ut
        traces = pd.concat(traces, axis=1, sort=False)
        return traces


# In[163]:


#Test Portfolio
if run_tests:
    test_portfolio = Portfolio(my_stock_desc, my_unit_trust_desc, 'SGD')
    test_cur_assets = test_portfolio.asset_distibution()
    print(test_cur_assets)
    test_traces = test_portfolio.asset_trace()
    test_traces.plot()


# In[164]:


#Run
currency = 'SGD'
portfolio = Portfolio(my_stock_desc, my_unit_trust_desc, currency)
cur_assets = portfolio.asset_distibution()
total_invested = cur_assets['invested'].sum()
total_current = cur_assets['value'].sum()
total_per_change = (total_current - total_invested) / total_invested * 100.0

asset_history = portfolio.asset_trace()
asset_total_history = asset_history.sum(axis=1).to_frame()
asset_total_history.columns=['Current']

investment_history = portfolio.investment_trace()
investment_total_history = investment_history.sum(axis=1).to_frame()
investment_total_history.columns=['Invested']

gains = pd.concat([asset_total_history, investment_total_history], axis =1)


# In[165]:


#Display
print('----------------------------------------------------------------------------')
print('Asset summary')
print('----------------------------------------------------------------------------')
print(cur_assets[['stock','value','invested', '% change']])
print('----------------------------------------------------------------------------')
print('Total invested assets: {0:1.0f} {1}'.format(total_invested, currency))
print('Total current assets: {0:1.0f} {1}'.format(total_current, currency))
print('Total % change: {0:1.2f} '.format(total_per_change))
print('----------------------------------------------------------------------------')
fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize=(14,22), facecolor='w') #gridspec_kw =  dict(hspace=0.3),figsize=(12,9)
plt.subplots_adjust(hspace=0.3)

#By sector
sector_pie = cur_assets.groupby(['sector']).sum().sort_values(['value'], ascending=False).plot.pie(y='value', autopct='%1.0f%%', ax=axes[0,0], title='By sector')
sector_pie.legend(bbox_to_anchor=(1.6, 0.5)) #loc="upper right"

#By geography
df_geo = pd.DataFrame(columns = ['stock', 'value', 'geography'])
geo_row = 0
for row in cur_assets.itertuples():
    for key, perc in row.geography.items():
        geo_val = row.value * perc
        df_geo.loc[geo_row] = list([row.stock, geo_val, key])
        geo_row = geo_row + 1
geo_pie = df_geo.groupby(['geography']).sum().sort_values(['value'], ascending=False).plot.pie(y='value', autopct='%1.0f%%', ax=axes[0,1], title='By geography')
geo_pie.legend(bbox_to_anchor=(1.6, 0.5)) #loc="upper right"

#By account type
sector_pie = cur_assets.groupby(['acc']).sum().sort_values(['value'], ascending=False).plot.pie(y='value', autopct='%1.0f%%', ax=axes[1,0], title='By account type')
sector_pie.legend(bbox_to_anchor=(1.6, 0.5)) #loc="upper right"
axes[1,1].axis('off')

#Value
cur_assets['value'].plot.pie(labels = cur_assets['stock'], autopct='%1.0f%%', ax=axes[2,0], title='Current value')
cur_assets['invested'].plot.pie(labels = cur_assets['stock'], autopct='%1.0f%%', ax=axes[2,1], title='Invested amount')
asset_history.plot(ax=axes[3,0], title='Stock value since purchase time')
gains.plot(ax=axes[3,1], title = 'Total assests current vs invested')
plt.show()


# In[ ]:


interested_dict = [
    {'symbol' : '1211.HK', 'sector':'EV'},
]


# In[ ]:


test_stock = Stock(interested_dict[2]['symbol'], interested_dict[2])
print(test_stock.business_summary())
test_history = test_stock.asset_trace(get_date('2015-01-01'), assign_name=False, smooth=8)
test_history.plot()
year_2020 = test_history.loc[test_history.index >= pd.to_datetime(get_date('2020-01-01'))]
year_2020_peak = year_2020['Close'].max()

cur_val = test_stock.current_value()
cur_perc_drop = (year_2020_peak - cur_val) / year_2020_peak * 100.0 

growth_before_peak_1 = test_history['2019-01-01']
growth_before_peak_1 = growth_before_peak_1

print("2020 peak = {0:1.0f}, cur = {1:1.0f}, per drop = {2:1.2f}%".format(year_2020_peak, cur_val, cur_perc_drop))


# In[ ]:


#Ticker test
symbol='1211.HK'
ticker = yf.Ticker(symbol)
assert ticker.history(period='1d')['Close'].size > 0, "Invalid symbol " + symbol

