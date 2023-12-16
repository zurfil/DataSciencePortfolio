import pandas as pd

df = pd.read_csv('orders.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Order Date'] = df['Order Date'].dt.strftime('%Y-%m-%d')

print(df.head(15))
df.to_csv('orders.csv', index=False)
