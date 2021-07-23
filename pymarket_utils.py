#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from forex_python.converter import CurrencyRates
from abc import ABC, abstractmethod


# In[2]:


#global functions
"""
price_chart = {
    'USD' : {'USD':1.0, 'SGD':1.41, 'HKD':7.75, 'INR':75.63},
    'SGD' : {'USD':0.71, 'SGD':1.0, 'HKD':5.49, 'INR':53.44},
    'HKD' : {'USD':0.13, 'SGD':0.18, 'HKD':1.0, 'INR':0},
    'INR' : {'USD':0.013, 'SGD':0.019, 'HKD':0, 'INR':1.0},
}
"""
currency_rates_now = CurrencyRates()

def currency_conv(value, from_cur, to_cur):
    factor = currency_rates_now.get_rate(from_cur, to_cur)
    return value * factor

#format : YYYY-MM-DD
def get_date(string):
    return datetime.date.fromisoformat(string)

