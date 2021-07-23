#!/usr/bin/env python
# coding: utf-8

# In[1]:


#All packages can be installed using: pip install <package-name>
#Additional pip install lxml
import numpy as np
import pandas as pd
import datetime
import os
from forex_python.converter import CurrencyRates
from abc import ABC, abstractmethod


# In[2]:


class UtTicker:
    def __init__(self, config):
        #read unit trust history table
        self.unit_trust_history = {}
        self._get_unit_trust_history_all(config)
    
    def _get_unit_trust_history_all(self, config_path):
        """
        config_file_format:
        Date,Close
        yyyy-mm-dd,float
        yyyy-mm-dd,float
        """
        for file in os.listdir(config_path):
            full_file = os.path.join(config_path, file)
            if os.path.isfile(full_file):
                assert file not in self.unit_trust_history, "Multiple config files exist for same unit trust symbol"
                values = pd.read_csv(full_file)
                assert 'Date' in values.columns, "Config file <Date> field missing. Ensure no space between headers"
                assert 'Close' in values.columns, "Config file <Close> field missing. Ensure no space between headers"
                values['Date'] = pd.to_datetime(values['Date'])
                values = values.set_index('Date')
                file_key = os.path.splitext(file)[0]
                self.unit_trust_history[file_key] = values
                
    def Ticker(self, symbol):
        assert symbol in self.unit_trust_history, "No unit trust record for " + symbol 
        return self.unit_trust_history[symbol]


# In[5]:


if __name__ == '__main__':
    ut = UtTicker('config/unit_trust')
    ticker = ut.Ticker('SG9999000343U')
    print(ticker)

