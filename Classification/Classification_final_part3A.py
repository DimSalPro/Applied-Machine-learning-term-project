import calendar
import datetime
import pandas as pd

# # read csv
original_data = pd.read_csv('train.csv', sep=',')

# # take prices before cleaning to get as much as possible
# # uses explode to correspond every price to each hotel id. Then we compute the mean for every hotel id that has
# # more than one prices and put the price for every hotel id in the original data frame
data_test = original_data.drop(
    ['session_id', 'user_id', 'step', 'timestamp', 'action_type', 'reference', 'platform', 'city', 'device',
     'current_filters'], axis=1)

data_test = data_test[data_test['impressions'].notna()]
data_test = data_test[data_test['prices'].notna()]

data_test['impressions'] = data_test['impressions'].apply(lambda x: x.split('|'))
data_test['prices'] = data_test['prices'].apply(lambda x: x.split('|'))

data_test = data_test.apply(pd.Series.explode)
data_test['prices'] = data_test['prices'].astype(int)
data_test = data_test.groupby('impressions').mean().reset_index()

original_data = pd.merge(original_data, data_test, left_on='reference', right_on='impressions', how='left')

# # Outlier detection with z-score method group by sessions

# We group the dataset based on session id and count the number of values that are included each session in a descending
# order ( for each feature would be the same). From this, we calculate the mean and the standard deviation of steps per
# session. We implement the z-score method Z= (X-mean)/std putting the results in a new column. We drop the lines whose
# z-score column values are more than a threshold ( how many standard deviations included). Then, we keep the values of
# the original dataset that have the same index as the grouped
df_temp = original_data.groupby(by='session_id').count().sort_values(by='step', ascending=False)

mean1 = df_temp['step'].mean()
print('The mean steps per session is:', mean1, '\n')

std1 = df_temp['step'].std()
print('The standard deviation of steps per session is:', std1, '\n')

threshold = 4

df_temp["z-score"] = (df_temp['step'] - mean1) / std1  # turn values to z values and add these in a z-score column

df_temp = df_temp[df_temp['z-score'] < threshold]  # drop lines with an exrteme number of steps

original_data = original_data[original_data['session_id'].isin(df_temp.index)].sort_values(
    by='step', ascending=True)  # make match with initial

print(len(original_data))

# # Outlier detection with z-score method group by users

# # We group the dataset based on user id and count the number of values that are corresponding to each user in
# # a descending order ( for each feature would be the same). From this, we calculate the mean and the standard
# # deviation of steps per user. We implement the z-score method Z= (X-mean)/std putting the results in a new column.
# # We drop the lines whose z-score column values are more than a threshold ( how many standard deviations included).
# # Then, we keep the values of the original dataset that have the same index as the grouped
df_temp2 = original_data.groupby(by='user_id').count().sort_values(by='action_type', ascending=False)

mean2 = df_temp2['step'].mean()
print('The mean steps per user is:', mean2, '\n')

std2 = df_temp2['step'].std()
print('The standard deviation of steps per user is:', std2, '\n')
threshold = 4

df_temp2["z-score"] = (df_temp2['step'] - mean2) / std2  # turn values to z values and add these in a z-score column
df_temp2 = df_temp2[df_temp2['z-score'] < threshold]  # drop lines with an extreme number of action types

original_data = original_data[original_data['user_id'].isin(df_temp2.index)]  # make match with the initial dataset

print(len(original_data))

# # We group the dataset based on user id and count the number of values that are corresponding to each user in a
# # descending order ( for each feature would be the same). From this, we calculate the mean and the standard deviation
# # of sessions per user. We implement the z-score method Z= (X-mean)/std putting the results in a new column.
# # We drop the lines whose z-score column values are more than a threshold ( how many standard deviations included).
# # Then, we keep the values of the original dataset that have the same index as the grouped

df_temp3 = original_data.groupby(by='user_id').count().sort_values(by='action_type', ascending=False)

mean3 = df_temp3['session_id'].mean()
print('The mean steps per session is:', mean3, '\n')

std3 = df_temp3['session_id'].std()
print('The standard deviation of steps per user is:', std3, '\n')
threshold = 4

df_temp3["z-score"] = (df_temp3[
                           'session_id'] - mean3) / std3  # turn values to z values and add these in a z-score column
df_temp3 = df_temp3[df_temp3['z-score'] < threshold]  # drop lines with an extreme number of action types

original_data = original_data[original_data['user_id'].isin(df_temp3.index)]  # make match with the initial dataset

# print(len(original_data))

# # start processing

# # missing data percentage for every feature
missing_data = pd.DataFrame({'percent_missing': original_data.isnull().sum()*100 / len(original_data)})
with pd.option_context('display.max_rows', None):
    print(missing_data.sort_values('percent_missing', ascending=False))

# # reset the index because we drop rows on the outlier detection
original_data = original_data.reset_index().drop('index', axis=1)

# # duplicate action type to use the interactions after we drop the data to create the unique interactions feature
original_data['action_type2'] = original_data['action_type']

# # action type mapped to 0 and 1. 1 for click out 0 for everything else
original_data['action_type'] = original_data['action_type'].replace('clickout item', 1)
original_data['action_type'] = original_data['action_type'].replace(r'[^\d]+', 0, regex=True)


# # this will be applied in a group by session data frame. for every session it creates a column which has 0 for every
# # time the action type doesnt change and it adds one for every change of the action type. we keep the min of this ,
# # which is the rows until the first change (if any changes), and outputs these rows
def f(group):
    index = min(group['action_type'].cumsum().searchsorted(1), len(group))

    return group.iloc[0:index + 1]


# # implementation reset to new index
original_data = original_data.groupby('session_id').apply(f)
original_data = original_data.rename(columns={'session_id': 'sess_id'})
original_data = original_data.reset_index()

# # series with different interactions for each session(count unique interaction)
interaction = original_data.groupby('session_id')['action_type2'].apply(lambda x: x.nunique())

# # drop columns that we will not further use
data = original_data.drop(
    ['user_id', 'reference', 'prices_x', 'impressions_x', 'impressions_y', 'platform', 'city', 'action_type2'], axis=1)

# # encode current filters to 0 and 1. o for all the nan and 1 for all that doesnt have nan
data['current_filters'] = data['current_filters'].fillna(0)
data['current_filters'] = data['current_filters'].replace(r'[^\d]+', 1, regex=True)

# # map timestamp and hours to working hours or no working hours
# # export date and time using slicing and then replace hours from 8 to 16 (1) and 0 for all the others
data['datetime(UTC)'] = pd.to_datetime(data['timestamp'], unit='s', utc=True)
data['datetime'] = data['datetime(UTC)'].astype(str)
data['time'] = data['datetime'].str[11:-6]
data['date'] = data['datetime'].str[:10]
data[['hour', 'min', 'sec']] = data['time'].str.split(':', expand=True)
data['hour'] = data['hour'].astype(int)

data['hour'] = data['hour'].replace([0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 22, 23], 1)
data['hour'] = data['hour'].replace([8, 9, 10, 11, 12, 13, 14, 15, 16], 0)
data.rename({'hour': 'time_of_day'}, axis=1, inplace=True)


# # function to find the day from the timestamp
def findday(date):
    born = datetime.datetime.strptime(date, '%Y-%m-%d').weekday()
    return calendar.day_name[born]


# # map days to 0 and 1 for weekend or not. then replace the findings with 0 for weekends and 1 for saturday sunday
data['day'] = data['date'].apply(lambda x: findday(x))
data['day'] = data['day'].replace('Sunday', 1)
data['day'] = data['day'].replace('Saturday', 1)
data['day'] = data['day'].replace(r'[^\d]+', 0, regex=True)

# # get the session duration for each session last value - first value
session_duration = data.groupby('session_id')['timestamp'].apply(lambda x: max(x) - min(x))

# # get the number of steps(max because they are in order) for each session
steps = data.groupby('session_id')['step'].apply(lambda x: max(x))

# # get the max value for action type (1 if there is a 1 or else 0) for each session
action_type = data.groupby('session_id')['action_type'].apply(lambda x: max(x))

# # same as before for the current filter
current_filters = data.groupby('session_id')['current_filters'].apply(lambda x: max(x))

# # time of day is the rounded avg cause if he spend more time from 0 category we want 0 or else 1(for when the use
# # is done near 16.00 or 8.00 that the label changes)
time_of_day = data.groupby('session_id')['time_of_day'].apply(lambda x: round(sum(x) / len(x)))

# # same thinking as before for the weekend days( for when the day changes)
weekday = data.groupby('session_id')['day'].apply(lambda x: round(sum(x) / len(x)))

# # if one session has many prices take the median (because we want to see what is the majority price he searchs)
prices = data.groupby('session_id')['prices_y'].apply(lambda x: x.median())

# # combine all these in a df with renamed columns
horizontal_stack = pd.concat([session_duration.rename('session_duration'), steps.rename('steps'),
                              weekday.rename('weekday'), current_filters.rename('current_filters'),
                              time_of_day.rename('time_of_day'), interaction.rename('interactions'),
                              prices.rename('prices'), action_type], axis=1)

# # missing data percentage to see nan in prices
missing_data = pd.DataFrame({'percent_missing': horizontal_stack.isnull().sum() * 100 / len(horizontal_stack)})
with pd.option_context('display.max_rows', None):
    print(missing_data.sort_values('percent_missing', ascending=False))

# # replace nan values on prices replace with median
horizontal_stack['prices'] = horizontal_stack['prices'].fillna(horizontal_stack['prices'].median())

print(horizontal_stack)

# # save to csv to use on the classification file
horizontal_stack.to_csv('Classification_Project.csv', index=False)
