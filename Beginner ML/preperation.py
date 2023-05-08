# https://realpython.com/pandas-groupby/
# https://realpython.com/python-data-cleaning-numpy-pandas/

import numpy as np 
import scipy.stats
import pandas as pd 
import matplotlib.pyplot as plt 
from datetime import datetime
import csv


# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output to 25
pd.set_option("display.max_rows", 25)

# read data
dtypes = {
        '@ID':'str',
        'DEBIT.AMOUNT':'float64',
        'CREDIT.ACCT.NO':'int64',
        'CREDIT.CURRENCY':'str',
        'TRANSACTION ORIGIN':'str',
        'overide':'str',
        'CUSTOMER.COUNTRY':'str',
        'MSG PATTERN':'category',
}

incoming = pd.read_csv(
    "incoming.csv",
    dtype=dtypes,
    usecols=list(dtypes) + ["PROCESSING.DATE","CREDIT.CUSTOMER"],
    parse_dates=["PROCESSING.DATE"],
    na_values='XX'
)

# data set memory size
##print(incoming.memory_usage(index=False, deep=True))

# remove rows without customer ID
incoming.dropna(subset=['CREDIT.CUSTOMER'], axis='index', inplace=True)

# fill blank overide
incoming['overide'].fillna('clean', inplace=True)

# fill blank ORDERING CUSTOMER COUNTRY
incoming['CUSTOMER.COUNTRY'].fillna('XX', inplace=True)
print(incoming.isnull().sum())

# convert CUSTOMER ID from float to integer
incoming['CREDIT.CUSTOMER'] = pd.to_numeric(incoming['CREDIT.CUSTOMER'], downcast='integer')

# convert CREDIT CURRENCY to numeric index
currency = incoming['CREDIT.CURRENCY'].unique()
currency_dict = dict(zip(currency, range(len(currency))))
incoming['CREDIT.CURRENCY'] = incoming['CREDIT.CURRENCY'].map(currency_dict)

csv_file = "Currency.csv"
try:
    with open(csv_file, 'w') as f:
        for key in currency_dict.keys():
            f.write("%s,%s\n"%(key,currency_dict[key]))
except IOError:
    print("I/O error")


# print(currency)
# print(incoming[['PROCESSING.DATE', 'CREDIT.CURRENCY', 'overide']].head())

# convert TRANSACTION ORIGIN to numeric index
trns_cntry = incoming['TRANSACTION ORIGIN'].unique()
ordr_cntry = incoming['CUSTOMER.COUNTRY'].unique()
#union = [trns_cntry, ordr_cntry]
country = np.unique(np.concatenate((trns_cntry, ordr_cntry), axis=None))

country_dict = dict(zip(country, range(len(country))))


csv_file = "Country.csv"
try:
    with open(csv_file, 'w') as f:
        for key in country_dict.keys():
            f.write("%s,%s\n"%(key,country_dict[key]))
except IOError:
    print("I/O error")

# convert CUSTOMER.COUNTRY to numeric index
incoming['CUSTOMER.COUNTRY'] = incoming['CUSTOMER.COUNTRY'].map(country_dict)
incoming['TRANSACTION ORIGIN'] = incoming['TRANSACTION ORIGIN'].map(country_dict)


# convert OVERIDE to numeric index
overide_dict = {'backdated' : 0, 'inactive' : 1, 'incorrect' : 2, 'locked' : 3, 'over limit' : 4, 'suspect' : 5, 'clean' : 6}
incoming['overide'] = incoming['overide'].map(overide_dict)
csv_file = "Override.csv"
try:
    with open(csv_file, 'w') as f:
        for key in overide_dict.keys():
            f.write("%s,%s\n"%(key,overide_dict[key]))
except IOError:
    print("I/O error")

# split date
# add split year
incoming['PROCESSING YEAR'] = incoming.apply(lambda row: row['PROCESSING.DATE'].year, axis=1)
# add split month
incoming['PROCESSING MONTH'] = incoming.apply(lambda row: row['PROCESSING.DATE'].month, axis=1)
# add split day
incoming['PROCESSING DAY'] = incoming.apply(lambda row: row['PROCESSING.DATE'].day, axis=1)

incoming.to_csv('adjusted_incoming.csv', index=False)