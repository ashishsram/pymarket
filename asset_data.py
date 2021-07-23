#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from abc import ABC, abstractmethod

#My modules
import pymarket_utils as utils


# In[ ]:


#my_stock_data = [
#    {'symbol' : '1211.HK','purchase_date': utils.get_date('2019-01-01'),'purchase_price': 40.00   ,'quantity': 500},
#]
#
#my_unit_trust_data = [
#    {'symbol':'SG9999000343U','purchase_date': utils.get_date('2019-06-01'),'purchase_price':1.00,'quantity':1000},
#]


# In[ ]:


class AssetData:
    def __init__(self, symbol):
        self.symbol = symbol
        self.orders = pd.DataFrame(columns=['purchase_date','purchase_price','quantity'])
        
    def add_order(self, purchase_date, purchase_price, quantity):
        i = len(self.orders.index)
        self.orders.loc[i] = list([purchase_date, purchase_price, quantity])
        
    def total_quantity(self):
        return sum(self.orders['quantity'])
    
    def total_invested(self):
        return sum(self.orders['quantity'] * self.orders['purchase_price'])
    
    def _get_trace(self, start_date, end_date):
        """
        Trace must have range [start_date, end_date]
        """
        trace = self.orders.set_index('purchase_date')
        idx = pd.date_range(start_date, end_date)
        trace = trace.reindex(idx)
        trace = trace.fillna(0) #fill 0s for days with no orders
        return trace
    
    def trace_quantity(self, start_date, end_date):
        trace = self._get_trace(start_date, end_date)
        trace = trace.cumsum()['quantity'].to_frame()
        trace.columns = ['Quantity']
        return trace
    
    def trace_invested(self, start_date, end_date):
        trace = self._get_trace(start_date, end_date)
        trace = trace['purchase_price'] * trace['quantity']
        trace = trace.cumsum()
        return pd.DataFrame(trace, columns=['Value'])


# In[ ]:


class MyAssetDataDict:
    def __init__(self, config_path):
        self.assets = {}
        my_stock_data = pd.read_csv(config_path + 'my_holdings_stocks.csv', parse_dates=['purchase_date'])
        for order in my_stock_data.to_dict(orient='records'):
            order['purchase_date'] = order['purchase_date'].to_pydatetime()
            print(order)
            self._add_asset(order)
            
        my_unit_trust_data = pd.read_csv(config_path + 'my_holdings_ut.csv', parse_dates=['purchase_date'])
        for order in my_unit_trust_data.to_dict(orient='records'):
            order['purchase_date'] = order['purchase_date'].to_pydatetime()
            self._add_asset(order)
                
    def _add_asset(self, order):
        sym = order['symbol']
        if sym not in self.assets:
            self.assets[sym] = AssetData(order['symbol'])
        self.assets[sym].add_order(order['purchase_date'], order['purchase_price'], order['quantity'])
            
    def asset(self, symbol):
        assert symbol in self.assets, "No asset entry exists for symbol " + symbol
        return self.assets[symbol]


# In[ ]:


if __name__ == '__main__':
    config_path = '/mnt/c/ubuntu/my_assets/config/'
    mydatadict = MyAssetDataDict(config_path)
    asset = mydatadict.asset('TSLA')
    #print(asset.orders)
    #print(asset.total_quantity())
    #print(asset.total_invested())
    print(asset.trace_quantity(utils.get_date('2019-01-01'), pd.to_datetime('today')))
    print(asset.trace_invested(utils.get_date('2019-01-01'), pd.to_datetime('today')))

