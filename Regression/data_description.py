import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def outlier_detection(col, data_frame, th):
    # Insert column name and threshhold
    mean = data_frame[col].mean()
    std = data_frame[col].std()
    z_name = f'{str(col)}_z_score'
    data_frame[z_name] = (data_frame[col] - mean) / std
    data_frame = data_frame[data_frame[z_name] < th]
    return data_frame


def z_scores(col, data_frame):
    mean = data_frame[col].mean()
    std = data_frame[col].std()
    z_name = f'{str(col)}_z_score'
    data_frame[z_name] = (data_frame[col] - mean) / std
    # data_frame = data_frame[data_frame[z_name] < th]
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

print(df)
print(df.columns)

# Drop categories not suitable for the problem
drop_cars = ['Van/Minibus',
             'ΕπαγγελματικόEπιβατικό', 'Αγροτικό/Pickup',
             'Αγωνιστικό', 'ΤρέιλερΑυτοκινήτου']

# Create distribution plots for price, miles, year ,hp and cc features
for column in ['price', 'mileage', 'hp', 'cc', 'year']:
    df2 = z_scores(column, df)

# Remove outliers using Z-score
for column in ['price', 'mileage', 'hp', 'cc']:
    df = outlier_detection(column, df, 3)

# remove unrealistic input for car specs
df = df[df.price > 1000]
df = df[df['hp'] > 50]
df = df[df['cc'] > 500]
df = df[df.year <= 21]
df = df[df['mileage'] > 1000]
df.year = df.year + 2000

# remove models with less than 50 listings in total
cars_per_model = df.groupby('maker').count().sort_values(by='year', ascending=False)['model']
model_to_drop = cars_per_model[cars_per_model < 100].index
df = df[~df['maker'].isin(model_to_drop)]

for column in ['price_z_score', 'mileage_z_score', 'hp_z_score', 'cc_z_score']:
    
    plt.figure(4, figsize=(10, 5))
    sns.histplot(df2[column])
    plt.title(f'Distribution_of_{column}_with_outliers')
    plt.savefig(f'description/Distribution_of_{column}_with_outliers.jpg', bbox_inches='tight')
    plt.show()

    plt.figure(5, figsize=(10, 5))
    sns.histplot(df[column])
    plt.title(f'Distribution of {column} without outliers')
    plt.savefig(f'description/Distribution_of_{column}_without_outliers.jpg', bbox_inches='tight')
    plt.show()

plt.figure(6, figsize=(10, 5))
car_types = df2.groupby('car_type').count().sort_values(by='maker',ascending=False)['maker'].plot.bar()
plt.title('Listings per car type')
plt.savefig('description/Listings_per_car_type.jpg', bbox_inches='tight')
plt.show()

plt.figure(7, figsize=(10, 5))
car_fuel = df2.groupby('car_fuel').count().sort_values(by='maker',ascending=False)['maker'].plot.bar()
plt.title('Listings per car_fuel')
plt.savefig(f'description/Listings_per_car_fuel.jpg', bbox_inches='tight')
plt.show()

plt.figure(8, figsize=(10, 5))
makers = df2.groupby('maker').count().sort_values(by='model',ascending=False)['model'].plot.bar()
plt.title('Listings per maker')
plt.savefig(f'description/Listings_per_maker.jpg', bbox_inches='tight')
plt.show()

plt.figure(9, figsize=(10, 5))
years = df2.groupby('year').count().sort_index()['model'].plot.bar()
plt.title('Listings per year')
plt.savefig(f'description/Listings_per_year.jpg', bbox_inches='tight')
plt.show()

# df.to_csv('cars__test_py.csv', index=False)
