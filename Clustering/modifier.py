import pandas as pd


def bits(df):
    min_8 = -128
    max_8 = 127
    min_16 = -32768
    max_16 = 32767
    min_32 = -2147483648
    max_32 = 2147483647

    # Loop through columns of DF
    for col in df.columns:

        try:
            min_ = df[col].min()
            max_ = df[col].max()
            type_ = df[col].dtypes
            ints = ['int64', 'int32', 'int16', 'int8']
            floats = ['float64', 'float32', 'float16']

            # If dtype is int compute min and max
            if type_ in ints:

                if df[col].dtypes == 'int64':
                    if min_ > min_8 and max_ < max_8:
                        df[col] = df[col].astype('int8')
                    elif min_ > min_16 and max_ < max_16:
                        df[col] = df[col].astype('int16')
                    elif min_ > min_32 and max_ < max_32:
                        df[col] = df[col].astype('int32')
                    else:
                        df[col] = df[col].astype('int64')

                elif df[col].dtypes == 'int32':
                    if min_ > min_8 and max_ < max_8:
                        df[col] = df[col].astype('int8')
                    elif min_ > min_16 and max_ < max_16:
                        df[col] = df[col].astype('int16')
                    else:
                        df[col] = df[col].astype('int32')

                elif df[col].dtypes == 'int16':
                    if min_ > min_8 and max_ < max_8:
                        df[col] = df[col].astype('int8')
                    else:
                        df[col] = df[col].astype('int16')
                else:
                    next
            else:
                next

        except:
            pass

    return df
