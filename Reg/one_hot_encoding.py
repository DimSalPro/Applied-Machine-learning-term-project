import pandas as pd



# create dummies
def dummies(data, col, drop=False):
    # i is optional
    # if one it will create dummies and remove last column
    # by default is set to 0 and it has no effect
    dummies_temp = pd.get_dummies(data[col],drop_first=drop)
    length = len(dummies_temp.columns)

    # dummies_temp = dummies_temp.iloc[:, :(length - i)]
    dataframe = pd.concat([data, dummies_temp], axis=1)
    dataframe.drop(col, axis=1, inplace=True)
    return dataframe


# read data
df = pd.read_csv('cars__test_py.csv')

# drop unused columns
df.drop(['price_z_score', 'mileage_z_score', 'hp_z_score',
         'cc_z_score','maker','car_type'], axis=1, inplace=True)

# One hot encoding for the specified columns in the list below
for col in ['car_fuel', 'transmission','model']:
    df = dummies(df, col)


df.to_csv('cars__test_py_encoded.csv', index=False)
