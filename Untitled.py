#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np


# In[62]:


ticker = "AAPL"
opening_price = 142.7
closing_price = 143.2
volume = 1200000
print(ticker, opening_price, closing_price, volume)


# In[63]:


Currency_pair = "EUR/USD"
Buying_rate = 1.1825
Seeling_rate = 1.1830
print(Currency_pair, Buying_rate, Seeling_rate)


# In[64]:


liste = ["AAPL", "MSFT", "GOOGL"]
liste.append("IBM")
print(liste)


# In[65]:


stock_details = {
"ticker": "AAPL",
"opening_price": 142.7,
"closing_price": 143.2,
"volume": 1200000
}
print(stock_details)


# In[66]:


bond_details = {
"Issuer": "Tom", 
"Maturity Date": "2023-12-01", 
"Coupon Rate": 4.2,
"Face Value": 120,
}
print(bond_details)


# In[67]:


stock_prices = [100, 101, 102, 98, 97]
for i in range(1, len(stock_prices)):
    daily_return = (stock_prices[i] - stock_prices[i-1]) / stock_prices[i-1]
    print(daily_return)


# In[68]:


principal = 1000
rate = 0.05 #% 5% annual interest
years = 0
while principal < 2000:
    principal *= (1 + rate)
    years += 1
    
print(years)


# In[69]:


stock_prices = [105, 107, 104, 106, 103]
test = []
for i in range(1, len(stock_prices)):
    daily_return = (stock_prices[i] - stock_prices[i-1]) / stock_prices[i-1]
    test.append(daily_return)
    print(daily_return)
    
print(np.mean(test))


# In[70]:


principal = 500
rate = 0.07
years = 0
while principal < 1000:
    principal *= (1 + rate)
    years += 1
print(years)
print(principal)


# In[71]:


bond_yield = 4.5
if bond_yield > 4.0:
    print("Buy the bond.")


# In[72]:


pe_ratio = 20
if pe_ratio < 15:
    print("Buy the stock.")
elif pe_ratio > 25:
    print("Sell the stock.")
else:
    print("Hold the stock.")


# In[73]:


bond_yield = 3.8
if bond_yield >= 4.0:
    print("Buy the bond.")
else:
    print("Do not buy the bond.")


# In[74]:


pe_ratio = 17
if pe_ratio < 15:
    print("Buy the stock.")
elif pe_ratio > 25:
    print("Sell the stock.")
else:
    print("Hold the stock.")


# In[75]:


pe_ratio = 15
if pe_ratio < 16 and pe_ratio > 14:
    print("Buy the stock.")
elif pe_ratio > 23 and pe_ratio < 27:
    print("Sell the stock.")
else:
    print("Hold the stock.")


# In[76]:


class Stock:
        def __init__(self, name, price, dividend):
            self.name = name
            self.price = price
            self.dividend = dividend
        def yield_dividend(self):
            return self.dividend / self.price

apple_stock = Stock('Apple', 150, 0.82)
google_stock = Stock('Google', 150, 0.82)
facebook_stock = Stock('Facebook', 150, 0.82)
print(apple_stock.yield_dividend())


# In[77]:


class Portfolio:
        def __init__(self, name):
            self.name = name
            self.instruments = []
                  
        def add_instrument(self, stock):
            self.instruments.append([stock.name, stock.price, stock.dividend])
        
        def total_value(self):
            test = 0
            for i in range(len(self.instruments)):
                test += self.instruments[i][1]
            return test

portfolio_1 = Portfolio('Portfolio_1')
portfolio_1.add_instrument(apple_stock)
portfolio_1.add_instrument(google_stock)
portfolio_1.add_instrument(facebook_stock)

print(portfolio_1.instruments)


# In[78]:


portfolio_1.total_value()


# In[79]:


class CurrencyConverter:
        def __init__(self):
            self.conversion_rate = {}
                  
        def add_rate(self, pair, rate):
            self.conversion_rate[pair] = rate
            
        def convert(self, amount, source_currency, target_currency):
            return amount * self.conversion_rate[source_currency + "/" + target_currency]
    
                


# In[80]:


currency_converter_1 = CurrencyConverter()
currency_converter_1.add_rate("EUR/USD", 1.05)
currency_converter_1.add_rate("USD/EUR", 0.95)
currency_converter_1.convert(100, "EUR", "USD")


# In[81]:


import numpy as np
prices = np.array([100, 102, 104, 101, 99, 98])
returns = (prices[1:] - prices[:-1]) / prices[:-1]
print("Daily returns:", returns)
annual_volatility = np.std(returns) * np.sqrt(252)
print("Annualized volatility:", annual_volatility)


# In[82]:


import numpy as np

np.random.seed(0)
daily_returns = np.random.normal(0.001, 0.02, 1000)
stock_prices = [100]
for r in daily_returns:
    stock_prices.append(stock_prices[-1] * (1+r))
    
stock_prices


# In[83]:


sigma_1 = 0.1
sigma_2 = 0.2

p1_2 = 0.5

w_1 = 0.6
w_2 = 0.4

var_p = w_1**2 * sigma_1**2 + w_2**2 * sigma_2**2 + 2 * w_1 * w_2 * sigma_1 * sigma_2 * p1_2
print(var_p)


# In[84]:


asset_A = [0.1, 0.2]
asset_B = [0.15, 0.3]

weights_A = np.array([i/10 for i in range(11)])

weights_B = 1 - weights_A

returns = asset_B[0]*weights_B + asset_A[0]*weights_A

print("returns :", returns)

covA_B = 0.5

vola = (weights_B**2 * asset_B[1]**2 + weights_A**2 * asset_A[1]**2 + 2 * weights_A * weights_B * asset_A[1] * asset_B[1] * covA_B)**0.5
print("volatilities :", vola)


# In[85]:


import matplotlib.pyplot as plt
import seaborn as sns
stock_prices = [100, 102, 104, 103, 105, 107, 108]
dates = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.figure(figsize=(10, 6))
sns.lineplot(x=dates, y=stock_prices)
plt.title('Stock Price Over a Week')
plt.xlabel("Days")
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()


# In[86]:


import matplotlib.pyplot as plt
stock_prices = [105, 103, 106, 109, 108, 107, 110, 112, 111, 113]
plt.plot(stock_prices)
plt.title("Stock Prices Over 10 Days")
plt.xlabel('Days')
plt.ylabel("Stock Price")
plt.show()


# In[87]:


import matplotlib.pyplot as plt
stock_prices_1 = [105, 103, 106, 109, 108, 107, 110, 112, 111, 113]
stock_prices_2 = [107, 108, 107, 107, 106, 108, 109, 108, 109, 110]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Days')
ax1.set_ylabel('stock price 1', color='red')
ax1.plot(stock_prices_1, color='red', ls="solid")
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('stock price 2', color='blue')
ax2.plot(stock_prices_2, color='blue', ls="dotted")
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Stock Prices Over 10 Days')

plt.show()


# In[88]:


import matplotlib.pyplot as plt
import seaborn as sns

returns = [0.05, -0.02, 0.03, -0.01, 0.02, 0.03, -0.03, 0.01, 0.04, -0.01]
sns.histplot(returns, bins=5)
plt.title("Distribution of Stock Returns")
plt.show()


# In[89]:


import matplotlib.pyplot as plt
import seaborn as sns

returns = [0.05, -0.02, 0.03, -0.01, 0.02, 0.03, -0.03, 0.01, 0.04, -0.01]        
sns.histplot(returns, bins=5, kde=True)
plt.title("Distribution of Stock Returns")
plt.show()


# In[90]:


import matplotlib.pyplot as plt
import seaborn as sns

returns = np.random.normal(0, 1, 10000)
sns.histplot(returns, bins=50, kde=True)
plt.title("Distribution of Stock Returns")
plt.show()


# In[ ]:





# In[98]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#finance classes
class Stock:
        def __init__(self, name, price, dividend):
            self.name = name
            self.price = price
            self.dividend = dividend
        def yield_dividend(self):
            return self.dividend / self.price

apple_stock = Stock('Apple', 120, 0.82)
google_stock = Stock('Google', 150, 0.82)
facebook_stock = Stock('Facebook', 130, 0.82)

print("Apple stock yield dividend: ", apple_stock.yield_dividend())

class Portfolio:
        def __init__(self, name):
            self.name = name
            self.instruments = []
                  
        def add_instrument(self, stock):
            self.instruments.append([stock.name, stock.price, stock.dividend])
        
        def total_value(self):
            test = 0
            for i in range(len(self.instruments)):
                test += self.instruments[i][1]
            return test

portfolio_1 = Portfolio('Portfolio_1')
portfolio_1.add_instrument(apple_stock)
portfolio_1.add_instrument(google_stock)
portfolio_1.add_instrument(facebook_stock)

print("Portfolio instruments: ", portfolio_1.instruments)
print("Portfolio total value: ", portfolio_1.total_value())

class CurrencyConverter:
        def __init__(self):
            self.conversion_rate = {}
                  
        def add_rate(self, pair, rate):
            self.conversion_rate[pair] = rate
            
        def convert(self, amount, source_currency, target_currency):
            return amount * self.conversion_rate[source_currency + "/" + target_currency]
    
currency_converter_1 = CurrencyConverter()
currency_converter_1.add_rate("EUR/USD", 1.05)
currency_converter_1.add_rate("USD/EUR", 0.95)
print("100 euros to usd: ", currency_converter_1.convert(100, "EUR", "USD"))

#2-assets portfolio returns and volatilities

asset_A = [0.1, 0.2]
asset_B = [0.15, 0.3]

weights_A = np.array([i/10 for i in range(11)])

weights_B = 1 - weights_A

returns = asset_B[0]*weights_B + asset_A[0]*weights_A

covA_B = 0.5

vola = (weights_B**2 * asset_B[1]**2 + weights_A**2 * asset_A[1]**2 + 2 * weights_A * weights_B * asset_A[1] * asset_B[1] * covA_B)**0.5
print("returns :", returns)
print("volatilities :", vola)

#finance plots
stock_prices_1 = [105, 103, 106, 109, 108, 107, 110, 112, 111, 113]
stock_prices_2 = [107, 108, 107, 107, 106, 108, 109, 108, 109, 110]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Days')
ax1.set_ylabel('stock price 1', color='red')
ax1.plot(stock_prices_1, color='red', ls="solid")
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('stock price 2', color='blue')
ax2.plot(stock_prices_2, color='blue', ls="dotted")
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Stock Prices Over 10 Days')

plt.show()

returns = [0.05, -0.02, 0.03, -0.01, 0.02, 0.03, -0.03, 0.01, 0.04, -0.01]
sns.histplot(returns, bins=5)
plt.title("Distribution of Stock Returns")
plt.show()

#In[99]:
def calculate_average(prices):
    return sum(prices) / len(prices)
average_price = calculate_average(stock_prices)
print(f"Average Stock Price: ${average_price}")

#In[100]:
# Example time series data (replace this with your actual data)
stock_prices = [100, 110, 105, 120, 115]  # Replace this with your actual data

def find_highest_price_day(prices):
    max_price = max(prices)
    max_day = prices.index(max_price) + 1  # Adding 1 to get the day (assuming day 1 corresponds to index 0)
    return max_day

highest_price_day = find_highest_price_day(stock_prices)
print(f"Day with the highest stock price: Day {highest_price_day}")

#In[101]:
# Updated time series data
stock_prices = [100, 110, 105, 120, 115, 157, 152]  # Updated data

def analyze_stock_trend(prices):
    trend = None
    increasing_count = sum(prices[i] < prices[i + 1] for i in range(len(prices) - 1))
    decreasing_count = sum(prices[i] > prices[i + 1] for i in range(len(prices) - 1))
    
    if increasing_count > decreasing_count:
        trend = "increasing"
    elif increasing_count < decreasing_count:
        trend = "decreasing"
    else:
        trend = "stable"
    
    return trend

stock_trend = analyze_stock_trend(stock_prices)
print(f"The stock prices are generally {stock_trend}.")    
    
#In[102]:
import statistics
def calculate_volatility(prices):
    return statistics.stdev(prices)
volatility = calculate_volatility(stock_prices)
print(f"Volatility: ${volatility}")

#In[103]:
dates = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
stock_prices = [150, 152, 151, 153, 152]

def calculate_average(prices):
    return sum(prices) / len(prices)

def highlight_days_above_average(prices):
    average_price = calculate_average(prices)
    print(f"Average Stock Price: ${average_price}")
    
    for i, price in enumerate(prices):
        if price > average_price:
            print(f"{dates[i]} has a stock price (${price}) above the average.")
        else:
            print(f"{dates[i]} has a stock price (${price}) at or below the average.")

highlight_days_above_average(stock_prices)

#[In104]:
    # Function to forecast the next day's stock price based on the average increase or decrease of the previous days
def forecast_next_day_price(prices):
    if len(prices) < 2:
        return "Insufficient data to forecast"
    
    price_diff_sum = sum(prices[i + 1] - prices[i] for i in range(len(prices) - 1))
    avg_price_diff = price_diff_sum / (len(prices) - 1)
    next_day_forecast = prices[-1] + avg_price_diff
    
    return f"Forecast for the next day's stock price: ${next_day_forecast:.2f}"

# New set of stock prices (extend the original data)
new_stock_prices = [152, 155, 153, 158, 157]  # Add your own values here

forecast = forecast_next_day_price(new_stock_prices)
print(forecast)

#In[105]: 
def present_value(fv, r, n):
    return fv / (1 + r)**n
FV = 120 
r = 0.05 
n = 2
PV = present_value(FV, r, n)
print(f"The present value is: ${PV:.2f}")
    
#In[106]:
def present_value(fv, r, n):
    return fv / (1 + r)**n

FV_2 = 500  # Future value
r_2 = 0.06  # Interest rate
n_2 = 2     # Number of years

PV_2 = present_value(FV_2, r_2, n_2)
print(f"The present value is: ${PV_2:.2f}")

    
#In[107]:
def future_value(pv, r, n):
    return pv * (1 + r)**n
PV = 90 
r = 0.07 
n=1
FV = future_value(PV, r, n)
print(f"The future value is: ${FV:.2f}")

#In[108]:
def future_value(pv, r, n):
        return pv * (1 + r)**n

PV_2 = 200   # Present value
r_2 = 0.03   # Interest rate
n_2 = 2      # Number of years

FV_2 = future_value(PV_2, r_2, n_2)
print(f"The future value is: ${FV_2:.2f}")
    
#In[109]:
PV_3 = 150   # Present value (initial investment)
r_3 = 0.05   # Interest rate
n_3 = 3      # Number of years

FV_3 = future_value(PV_3, r_3, n_3)
print(f"The future value is: ${FV_3:.2f}")
    
    
#In[110]:
def compound(pv, r):
     return pv * (1 + r)
PV = 80 
r = 0.09
FV = compound(PV, r)
print(f"After one year with a 9% interest rate, you’ll have: ${FV:.2f}")

#In[111]:
def present_value(fv, r):
    return fv / (1 + r)

FV_2 = 115  # Future value
r_2 = 0.08  # Interest rate

PV_2 = present_value(FV_2, r_2)
print(f"The present value is: ${PV_2:.2f}")
    
#In[112]:
def present_value_target(fv, r, n):
        return fv / (1 + r)**n

FV_3 = 500  # Future value
r_3 = 0.06  # Interest rate
n_3 = 2     # Number of years

PV_3 = present_value_target(FV_3, r_3, n_3)
print(f"To have $500 after two years, you need to invest: ${PV_3:.2f}")
    
#In[113]:
FV_4 = 180  # Future value
r_4 = 0.10  # Discount rate
n_4 = 2     # Number of years

PV_4 = present_value_target(FV_4, r_4, n_4)
print(f"The present value of the promise today is: ${PV_4:.2f}")

#In[114]:
FV_5 = 1000  # Future value
r_5 = 0.07   # Interest rate
n_5 = 3      # Number of years

PV_5 = present_value_target(FV_5, r_5, n_5)
print(f"To have $1000 after three years, you need to invest: ${PV_5:.2f}")
    
#In[115]:
import yfinance as yf
# Download Microsoft stock data
msft_data = yf.download("MSFT", start="2021-01-01", end="2022-01-01")
print(msft_data.head())
    

#In[116]:
# Retrieve Google stock data
googl_data = yf.download("GOOGL", start="2020-01-01", end="2022-12-31")
print(googl_data.head())
   
 #In[117]:
# Fetch Amazon stock data for the last quarter of 2021
amzn_data = yf.download("AMZN", start="2021-10-01", end="2021-12-31")
print(amzn_data.head())
    
#In[118]:
tesla_data = yf.download("TSLA", start="2020-01-01", end="2021-01-01")
tesla_data['Close'].plot(figsize=(10, 5))
plt.title('Tesla Stock Closing Prices 2020')
plt.ylabel( 'Price (in \$)') 
plt.xlabel( 'Date' )
plt.show()

#In[119]:
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch Netflix stock data for the first half of 2022
netflix_data = yf.download("NFLX", start="2022-01-01", end="2022-06-30")

# Plotting the daily closing prices
netflix_data['Close'].plot(figsize=(10, 5))
plt.title('Netflix Stock Closing Prices - First Half of 2022')
plt.ylabel('Price (in $)')
plt.xlabel('Date')
plt.show()


#In[120]:
# Fetch Facebook stock data for the entire year of 2019
facebook_data = yf.download("META", start="2019-01-01", end="2019-12-31")
# Plotting the daily closing prices
facebook_data['Close'].plot(figsize=(10, 5))
plt.title('Facebook Stock Closing Prices - 2019')
plt.ylabel('Price (in $)')
plt.xlabel('Date')
plt.show()

#In[121]:
ibm_data = yf.download("IBM", start="2020-01-01", end="2021-01-01")
ibm_data['30-day MA'] = ibm_data['Close'].rolling(window=30).mean()
ibm_data[['Close’, ’30-day MA']].plot(figsize=(10, 5))
plt.title('IBM Stock Prices with 30-day Moving Average 2020')
plt.ylabel('Price (in \$)')
plt.xlabel('Date')
plt.show()

#In[122]:
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch Adobe Systems stock data for 2021
adbe_data = yf.download("ADBE", start="2021-01-01", end="2021-12-31")

# Calculate 20-day moving average
adbe_data['20-day MA'] = adbe_data['Close'].rolling(window=20).mean()

# Plotting the daily closing prices with 20-day moving average
plt.figure(figsize=(10, 5))
plt.plot(adbe_data['Close'], label='Daily Closing Prices')
plt.plot(adbe_data['20-day MA'], label='20-day MA')
plt.title('Adobe Systems Stock Prices with 20-day Moving Average - 2021')
plt.xlabel('Date')
plt.ylabel('Price (in $)')
plt.legend()
plt.show()

#In[123]:
# Fetch Nvidia Corporation stock data for 2022
nvda_data = yf.download("NVDA", start="2022-01-01", end="2022-12-31")

# Calculate 40-day moving average
nvda_data['40-day MA'] = nvda_data['Close'].rolling(window=40).mean()

# Plotting the daily closing prices with 40-day moving average
plt.figure(figsize=(10, 5))
plt.plot(nvda_data['Close'], label='Daily Closing Prices')
plt.plot(nvda_data['40-day MA'], label='40-day MA')
plt.title('Nvidia Corporation Stock Prices with 40-day Moving Average - 2022')
plt.xlabel('Date')
plt.ylabel('Price (in $)')
plt.legend()
plt.show()        
    
#In[124]:
sbux_data = yf.download("SBUX", start="2020-01-01", end="2021-01-01")
monthly_data = sbux_data['Close'].resample('M').mean()
monthly_data.plot(figsize=(10, 5))
plt.title('Starbucks Monthly Average Closing Prices 2020')
plt.ylabel('Price (in \$)')
plt.xlabel('Date')
plt.show()
    
#In[125]:
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch Disney stock data for 2019
dis_data = yf.download("DIS", start="2019-01-01", end="2020-01-01")

# Resample to bi-weekly (every two weeks) average closing prices
biweekly_data = dis_data['Close'].resample('2W').mean()

# Plotting bi-weekly average closing prices
plt.figure(figsize=(10, 5))
biweekly_data.plot()
plt.title('Disney Bi-Weekly Average Closing Prices - 2019')
plt.xlabel('Date')
plt.ylabel('Price (in $)')
plt.show()

#In[126]:
# Fetch Coca-Cola Company stock data for 2020
ko_data = yf.download("KO", start="2020-01-01", end="2021-01-01")

# Resample to quarterly average closing prices
quarterly_data = ko_data['Close'].resample('Q').mean()

# Plotting quarterly average closing prices
plt.figure(figsize=(10, 5))
quarterly_data.plot()
plt.title('Coca-Cola Company Quarterly Average Closing Prices - 2020')
plt.xlabel('Date')
plt.ylabel('Price (in $)')
plt.show()
    
    