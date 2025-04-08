import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random

def fetch_dataset(data_dir):
    dataframes = {}  # Dictionary to store the dataframes with file name as the key
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):  # Ensure we only process CSV files
        
            base_name = file[:-4]        
            print(base_name)
            clean_df = pd.read_csv(os.path.join(data_dir, file))
            clean_df['Date'] = pd.to_datetime(clean_df['Date'], errors= 'coerce')
            clean_df = clean_df.dropna(subset=['Date'])
            
            # Extract year, month, day, hour, and minute components
            clean_df['Date'] = pd.to_datetime(clean_df['Date'], errors= 'coerce')
            clean_df['Year'] = clean_df['Date'].dt.year
            clean_df['Month'] = clean_df['Date'].dt.month
            clean_df['Day'] = clean_df['Date'].dt.day
            clean_df['Hour'] = clean_df['Date'].dt.hour
            clean_df['Minute'] = clean_df['Date'].dt.minute
        
            # set date as index column
            clean_df = clean_df.set_index('Date')
            # Leave columns with keyword of 'kW' and calculate sum of energy consumption of all floors 
            # tmp1 = clean_df.loc[:, clean_df.columns.str.contains('kW')].groupby('Date').sum()
            tmp1 = clean_df.loc[:, clean_df.columns.str.contains('kW')].groupby('Date').sum()

            # sum of energy consumption of all power outlets in all floors and zones
            # tmp2 = tmp1.sum(axis=1).rename('total_demand').to_frame()
            
            # df_powerMeter = pd.concat([tmp2, tmp1], axis=1)
            tmp2 = tmp1.sum(axis=1).rename('total_demand').to_frame()
            tmp3 = clean_df.loc[:, clean_df.columns.str.contains('AC')].groupby('Date').sum().sum(axis=1).rename('AC').to_frame()
            tmp4 = clean_df.loc[:, clean_df.columns.str.contains('Light')].groupby('Date').sum().sum(axis=1).rename('Light').to_frame()
            tmp5 = clean_df.loc[:, clean_df.columns.str.contains('Plug')].groupby('Date').sum().sum(axis=1).rename('Plug').to_frame()

            df_powerMeter = pd.concat([tmp2, tmp1, tmp3,tmp4,tmp5], axis=1)

            # mean of each T mins
            df_powerMeter = df_powerMeter.resample('10T').mean()

            # rows_to_exclude = random.randint(0, int(len(df_powerMeter)*0.4))  # Up to 40% of rows
            # if rows_to_exclude > 0:
            #     df_powerMeter = df_powerMeter[:-rows_to_exclude]
            # Extract the date components
            df_powerMeter['Year'] =  pd.DatetimeIndex(df_powerMeter.index).year
            df_powerMeter['Month'] = pd.DatetimeIndex(df_powerMeter.index).month
            df_powerMeter['Day'] = pd.DatetimeIndex(df_powerMeter.index).day
            df_powerMeter['Hour'] = pd.DatetimeIndex(df_powerMeter.index).hour
            df_powerMeter['Minute'] = pd.DatetimeIndex(df_powerMeter.index).minute
    
            df_powerMeter.index = pd.to_datetime(df_powerMeter.index, errors='coerce')
            # df_powerMeter = df_powerMeter.dropna(subset = ['z1_AC1(kW)', 'z1_AC2(kW)', 'z1_AC3(kW)', 'z1_AC4(kW)', 'z1_Light(kW)', 'z1_Plug(kW)', 'z2_AC1(kW)', 'z2_Light(kW)', 'z2_Plug(kW)', 'z3_Light(kW)', 'z3_Plug(kW)','z4_AC1(kW)', 'z4_Light(kW)', 'z4_Plug(kW)', 'z5_AC1(kW)', 'z5_Light(kW)', 'z5_Plug(kW)'], how='all')
            df_powerMeter = df_powerMeter[['total_demand','AC', 'Light', 'Plug', 'Year', 'Month', 'Day', 'Hour', 'Minute']]
            df_powerMeter = df_powerMeter.dropna(how='any')
            
            df_powerMeter.sort_values('Date')
            
            dataframes[base_name] = df_powerMeter
            
    return dataframes 