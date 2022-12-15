import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def outlier_detection(col, data_frame, th):
    # Insert column name and threshhold
    mean = data_frame[col].mean()
    std = data_frame[col].std()
    z_name = f'{str(col)}_z_score'
    data_frame[z_name] = (data_frame[col] - mean) / std
    data_frame = data_frame[data_frame[z_name] < th]
    return data_frame


# Read csv 1,2,3 and create listing date column based on extraction date
df_1 = pd.read_csv('cars_all_v3.csv')
df_1['listing_date'] = 1

df_2 = pd.read_csv('cars_all_v5.csv')
df_2['listing_date'] = 2

df_3 = pd.read_csv('cars_all_v7.csv')
df_3['listing_date'] = 3

# Drop duplicates from the new dataframe based on date
df = pd.concat([df_1, df_2, df_3])
df.drop_duplicates('link', keep='last', inplace=True)
df.drop(['listing_date', 'link'], axis=1, inplace=True)

# Split engine to cc and hp fields
df[['cc', 'hp']] = df['engine'].str.split('/', expand=True)
df.drop('engine', axis=1, inplace=True)

# Transform text to numerical
# Price
df['price'] = df['price'].astype(str).replace(r'[^\d]', '', regex=True)
df['price'] = df['price'].str[:7].astype(int)

# Mileage
df['mileage'] = df['mileage'].astype(str).apply(lambda x: x.replace('.', '')).astype(int)
# CC
df['cc'] = df['cc'].astype(int)
# HP
df = df[~df['hp'].isna()]
df['hp'] = df['hp'].replace(r'[^\d]', '', regex=True).astype(int)

# Transform categorical attributes
# Car_fuel
df.car_fuel = df.car_fuel.replace('Βενζίνη', 'gasoline')
df.car_fuel = df.car_fuel.replace('Πετρέλαιο', 'diesel')
df.car_fuel = df.car_fuel.replace('Αέριο(lpg)-Βενζίνη', 'gas')
df.car_fuel = df.car_fuel.replace('ΥβριδικόΒενζίνη', 'hybrid')
df.car_fuel = df.car_fuel.replace('Ηλεκτρικό', 'electric')
df.car_fuel = df.car_fuel.replace('ΥβριδικόΠετρέλαιο', 'hybrid')
df.car_fuel = df.car_fuel.replace('Φυσικόαέριο(cng)', 'gas')
df.car_fuel = df.car_fuel.replace('Αλλο', 'other')
df.car_fuel = df.car_fuel.replace('Φυσικόαέριο(cng)-Βενζίνη', 'gas')
# Drop cars with car_fuel type = other
df = df[df['car_fuel'] != 'other']
# Transmission
df.transmission = df.transmission.replace('Αυτόματο', 'auto')
df.transmission = df.transmission.replace('Χειροκίνητο', 'manual')
df.transmission = df.transmission.replace('Ημιαυτόματο', 'auto')

# Drop categories not suitable for the problem
drop_cars = ['Van/Minibus',
             'ΕπαγγελματικόEπιβατικό', 'Αγροτικό/Pickup',
             'Αγωνιστικό', 'ΤρέιλερΑυτοκινήτου']

for col in drop_cars:
    df=df[df['car_type'] != col]


# Remove outliers using Z-score
for column in ['price', 'mileage', 'hp', 'cc']:
    df = outlier_detection(column, df, 3)


# remove unrealistic input for car specs
df = df[df.price > 1000]
df=df[df['mileage']>1000]
# df = df[df.price>14000]
df = df[df['hp'] > 50]
df = df[df['cc'] > 500]
df = df[df.year <= 21]



# If you need to see different kind of results, uncomment the following lines of code for deeper data cleaning ( Line 94 to the end)

# remove models with less than 100 listings in total
# cars_per_model = df.groupby('model').count().sort_values(by='year', ascending=False)['maker']
# model_to_drop = cars_per_model[cars_per_model < 50].index
# df = df[~df['model'].isin(model_to_drop)]



# for model in df.model.unique():
#     brand = df[df['model']==model].groupby('maker').count().sort_values(by='model',ascending=False).head(1).index[0]
#     # c_type =  df[df['model']==model].groupby('car_type').count().sort_values(by='maker',ascending=False).head(1).index[0]
    
#     df['maker'] = np.where((df.model == model),brand,df.maker)
#     # df['car_type'] = np.where((df.model == model),c_type,df.car_type)

print(df)

df.to_csv('cars__test_py.csv', index=False)
