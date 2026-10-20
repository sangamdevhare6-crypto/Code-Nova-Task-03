"""
Script to generate the synthetic sales_data.csv dataset.
This is a helper script - the dataset is also committed directly.
"""
import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

stores = ['Store_A', 'Store_B', 'Store_C', 'Store_D']
products = ['Electronics', 'Clothing', 'Groceries', 'Furniture', 'Sports']

dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')

records = []
for date in dates:
    for store in random.sample(stores, k=random.randint(2, 4)):
        product = random.choice(products)
        price = round(random.uniform(10, 500), 2)
        ad_spend = round(random.uniform(50, 2000), 2)
        promotion = random.choice([0, 1])
        holiday = 1 if date.weekday() >= 5 else 0
        base_sales = (price * 0.8 + ad_spend * 0.15 + promotion * 200 + holiday * 150
                      + random.gauss(0, 50))
        sales = max(0, round(base_sales, 2))
        records.append([date.strftime('%Y-%m-%d'), store, product,
                        price, ad_spend, promotion, holiday, sales])

df = pd.DataFrame(records, columns=[
    'Date', 'Store', 'Product', 'Price',
    'Advertising_Spend', 'Promotion', 'Holiday', 'Sales'
])

# Introduce ~2% missing values
for col in ['Price', 'Advertising_Spend', 'Store', 'Product']:
    idx = df.sample(frac=0.02, random_state=42).index
    df.loc[idx, col] = np.nan

# Introduce ~1% duplicates
dup_rows = df.sample(frac=0.01, random_state=7)
df = pd.concat([df, dup_rows], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs('data', exist_ok=True)
df.to_csv('data/sales_data.csv', index=False)
print(f"Dataset generated: {len(df)} rows")
