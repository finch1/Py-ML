# https://realpython.com/pandas-groupby/
# https://realpython.com/python-histograms/
# https://realpython.com/python-matplotlib-guide/

# https://mode.com/blog/bridge-the-gap-window-functions?utm_medium=recommended&utm_source=blog&utm_content=thinking_python
# https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea
# https://towardsdatascience.com/5-minute-guide-to-plotting-with-pandas-e8c0f40a1df4

import numpy as np 
import scipy.stats
import pandas as pd 
import matplotlib.pyplot as plt 
from datetime import datetime
import csv
import plotly.graph_objs as go 

# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output to 25
pd.set_option("display.max_rows", 25)

# read data
# read data
dtypes = {
            #'CREDIT_ACCT_NO':'int64',             'COUNTRY_RISK_RATING_TOTAL':'float64',            
            #'ORDERING_CUST_COUNTRY':'int64',       #'INSITITUTION.ACCOUNT.COUNTRY':'int64',            
            # 'LOC_AMT_CREDITED':'int64',            'MSG_PATTERN':'category',
            #'OVERRIDE_SHORT':'int64',            
            #'PROCESSING_DATE_WD':'int16',    
            'CUSTOMER':'int64',            
            'DEBIT_AMOUNT':'float64',
            'DEBIT_CURRENCY':'str',
            'PROCESSING_DATE_MONTH':'int16',                  
            'PROCESSING_DATE_YEAR':'int16'
}

incoming = pd.read_csv('incoming_data.csv', dtype=dtypes, parse_dates=['PROCESSING_DATE'])
field_value = 101872
slider_value = [2019, 2019]
    
tra_per_ord = incoming.loc[(incoming.CUSTOMER == int(field_value))].groupby(
['VALUE_NAME', 'ORDERING_CUST_COUNTRY']).size().reset_index().sort_values([0], ascending=True)
print(tra_per_ord)

tra_per_cust = incoming.loc[(incoming.CUSTOMER == int(field_value))].groupby(['ORDERING_IBAN_COUNTRY', 'ORDERING_IBAN']).agg(
    {'LOC_AMT_CREDITED':['sum','count']}).reset_index()
print(tra_per_cust)
tra_per_cust.sort_values(by=['ORDERING_IBAN_COUNTRY', ('LOC_AMT_CREDITED', 'count')], ascending=True, inplace=True)      
print(tra_per_cust)

for label, label_df in tra_per_cust.groupby('ORDERING_IBAN_COUNTRY'):
        print(label_df.ORDERING_IBAN)
        print(label_df.LOC_AMT_CREDITED['count'])


print(tra_per_cust)

tra_per_cust = incoming.loc[(incoming.CUSTOMER == int(field_value)) & (incoming.PROCESSING_DATE_YEAR >= slider_value[0]) & (incoming.PROCESSING_DATE_YEAR <= slider_value[1])].groupby(
    ['PROCESSING_DATE_MONTH', 'DEBIT_CURRENCY']).agg({'DEBIT_AMOUNT': ['sum', 'count'], 'LOC_AMT_CREDITED': ['sum']}).reset_index().sort_values('PROCESSING_DATE_MONTH', ascending=True) #sql sum to pandas dataframe

df = pd.DataFrame(columns = ['PROCESSING_DATE_MONTH', 'DEBIT_CURRENCY', 'DEBIT_AMOUNT_sum', 'DEBIT_AMOUNT_count', 'LOC_AMT_CREDITED_sum']) 

df['PROCESSING_DATE_MONTH'] = tra_per_cust['PROCESSING_DATE_MONTH']
df['DEBIT_CURRENCY'] = tra_per_cust['DEBIT_CURRENCY']
df['DEBIT_AMOUNT_sum'] = tra_per_cust['DEBIT_AMOUNT']['sum']
df['DEBIT_AMOUNT_count'] = tra_per_cust['DEBIT_AMOUNT']['count']
df['LOC_AMT_CREDITED_sum'] = tra_per_cust['LOC_AMT_CREDITED']['sum']

hover_text = []
bubble_size = []
max_amnt = df['LOC_AMT_CREDITED_sum'].max()
min_amnt = df['LOC_AMT_CREDITED_sum'].min()
low_upp = [5, 15]

for index, row in df.iterrows():
    hover_text.append(('Currency: {currency}<br>'+
                        'Value: {value}<br>'+
                        'Month: {Month}').format(currency=row['DEBIT_CURRENCY'], # .values[0] values converts to list and then get first element
                                                value=row['DEBIT_AMOUNT_sum'],
                                                Month=row['PROCESSING_DATE_MONTH']))
    bubble_size.append(int(((((row['LOC_AMT_CREDITED_sum']-min_amnt)*(low_upp[1]-low_upp[0]))/(max_amnt-min_amnt))+low_upp[0]))) # normalize for bubble size. https://www.youtube.com/watch?v=SrjX2cjM3Es

df['text'] = hover_text
df['size'] = bubble_size





sizeref = 2.*max(df['size'])/(100**2)





for currency_name, currency in currency_data.items():
    print( currency_name,' ',currency)

#print(tra_per_ord[0])
print(tra_per_cust.head(10))
print(tra_per_cust.shape)

'''

## aggregation
dff = incoming.loc[(incoming.CREDIT_CUSTOMER == int(field_value)) & (incoming.PROCESSING_DATE_YEAR == int(2015))].groupby(
        'PROCESSING_DATE_MONTH').agg({'LOC_AMT_DEBITED': ['min', 'max', 'mean', 'median', 'sum']}).reset_index() #sql sum to pandas dataframe


## box plot
tra_per_ord = incoming.loc[(incoming.CREDIT_CUSTOMER == int(100223)) & (incoming.PROCESSING_DATE_YEAR == int(2019)) & (incoming.KEY_NAME == 'DETUR AUSTRIA GMBH')][['KEY_NAME','LOC_AMT_DEBITED']].reset_index() #sql sum to pandas dataframe


# data set memory size
# print(incoming.memory_usage(index=False, deep=True))

# select couple of rows
# print(incoming[['PROCESSING DAY', 'PROCESSING MONTH', 'PROCESSING YEAR','PROCESSING.DATE']])

# aggregate
# print(incoming['CREDIT.CUSTOMER'].mode())

# conditional 
# df.loc[df['shield'] > 6, ['max_speed']]

# print to csv
# days_acc.to_csv(r'pandas.txt', header=None, index=None, sep=' ', mode='a')

# print column types
# print(incoming.types)
'''
'''
## list accounts per customer
q1 = incoming.loc[incoming['CREDIT_CUSTOMER'] == 100144, ['ACCT_NO']]

## count number of accounts
## q2 = incoming.groupby(incoming['CREDIT_CUSTOMER', 'ACCT_NO'])[incoming['CREDIT_CUSTOMER']].apply(lambda x: (x==100144).sum()).reset_index(name='count')
# df11=df.groupby('key1')['key2'].apply(lambda x: (x=='one').sum()).reset_index(name='count')

## movement per day
q3 = incoming.loc[incoming['CREDIT_CUSTOMER'] == 100144, ['ACCT_NO', 'PROCESSING_DATE', 'DEBIT_AMOUNT']]


print(incoming[0:10])
# month,account 
month_acc = incoming.groupby(['CREDIT_CUSTOMER', 'ACCT_NO', 'PROCESSING_DATE_YEAR', 'PROCESSING_DATE_MONTH'])['DEBIT_AMOUNT'].sum()
month_acc_df = month_acc.to_frame()

# one_cus = month_acc_df.loc[100223]
one_cus = month_acc_df.loc[100223,12844625]
#print(one_cus)

# partition data 
'''
'''
popular_customer = incoming.loc[incoming['CREDIT.CUSTOMER'] == int(incoming['CREDIT.CUSTOMER'].mode())]

#incoming['DEBIT.AMOUNT'].plot()
incoming['DEBIT.AMOUNT'].plot.hist()
# group by month, year
trn_per_month = popular_customer.groupby(['PROCESSING YEAR', 'PROCESSING MONTH'])['PROCESSING MONTH'].count()
print(trn_per_month)

# group by currency
credit_currency = popular_customer.groupby(['CREDIT.CURRENCY'])['CREDIT.CURRENCY'].count()
# Generate data on commute times.
size, scale = 1000, 10

trn_per_month.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
plt.title('Commute Times for 1,000 Commuters')
plt.xlabel('Counts')
plt.ylabel('Commute Time')
plt.grid(axis='y', alpha=0.75)
plt.show() 

'''
'''

# bar chart month year

# box plot on amount
amount = np.array(df1['DEBIT.AMOUNT'])
fig, ax = plt.subplots()
ax.boxplot((amount), vert=False, showmeans=True, meanline=True, 
            patch_artist=True, medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()           

# histogram on amount
hist, bin_edges = np.histogram(amount, bins = 3)
fig, ax = plt.subplots()
ax.hist(amount, bin_edges, cumulative=False)
ax.set_xlabel('amount')
ax.set_ylabel('Frequency')
plt.show()

# pie chart currency
currency = np.array(df1['CREDIT.CURRENCY'])
fig, ax = plt.subplots()
ax.pie((currency))
plt.show()

# pie chart customer country
currency = np.array(df1['TRANSACTION ORIGIN'])
fig, ax = plt.subplots()
ax.pie((currency))
plt.show()

currency = np.array(df1['CUSTOMER.COUNTRY'])
fig, ax = plt.subplots()
ax.pie((currency))
plt.show()
'''