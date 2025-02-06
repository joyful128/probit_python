

import pandas as pd
import numpy as np
import tabula as tb
import statsmodels as sm

## Air Quality Variables ##
# Import Air Quality Data (Hourly Data by Measurement Station)

dataframes = []

file_paths = [
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_Jan.csv',
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_Feb.csv',
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_Mar.csv',
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_Apr.csv',
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_May.csv',
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_Jun.csv',
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_Jul.csv',
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_Aug.csv',
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_Sep.csv',
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_Oct.csv',
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_Nov.csv',
    '/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/2018/2018_Dec.csv'
]

# Combine and condition all Files

for file_path in file_paths :
    AQ_df = pd.read_csv(file_path,delimiter=';',na_values=['', ' ', 'NA', 'N/A', 'NULL'],keep_default_na=True)

    AQ_df['측정일시'] = pd.to_datetime(AQ_df['측정일시'], format='%Y%m%d%H', errors='coerce')
    AQ_df = AQ_df.dropna(subset=['측정일시'])
    AQ_df['date'] = AQ_df['측정일시'].dt.date

    for col in ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']:
        AQ_df[col] = pd.to_numeric(AQ_df[col].astype(str).str.replace(',', '.'), errors='coerce')

    dataframes.append(AQ_df)

AQ_combined = pd.concat(dataframes, ignore_index=True)

# Rename to English
AQ_combined.rename(columns={
    '지역': 'Region',
    '망': 'Network Type',
    '측정소코드': 'Station Code',
    '측정소명': 'Station Name',
    '측정일시': 'Measurement DateTime',
    'SO2': 'SO2',
    'CO': 'CO',
    'O3': 'O3',
    'NO2': 'NO2',
    'PM10': 'PM10',
    'PM25': 'PM2.5',
    '주소': 'Address'
}, inplace=True)


# Filter data points with at least 18 hours of data per day
AQ_group = AQ_combined.groupby(['Station Code', 'date']).filter(
    lambda x: (x['Network Type'].iloc[0] == '도시대기') & (len(x) >= 18)
)

# Average Data
AQ_daily = AQ_group.groupby(['Region','Station Code','date']).agg(
    SO2_avg=('SO2', 'mean'),
    CO_avg=('CO', 'mean'),
    O3_avg=('O3', 'mean'),
    NO2_avg=('NO2', 'mean'),
    PM10_avg=('PM10', 'mean'),
    PM2_5_avg=('PM2.5', 'mean')
).reset_index()

AQ_daily_avg = AQ_daily.groupby('Station Code').filter(lambda x: len(x) >= 365)


# Exceedance Value of Air Quality
AQ_daily_avg['PM2_5_exceed'] = AQ_daily_avg['PM2_5_avg'] > 15
AQ_daily_avg['PM10_exceed'] = AQ_daily_avg['PM10_avg'] > 45

AQ1 = AQ_daily_avg.groupby(['Region','Station Code']).agg(
    PM2_5_exceed_count=('PM2_5_exceed', 'sum'),
    PM10_exceed_count=('PM10_exceed', 'sum'),
    PM10_avg=('PM10_avg', 'mean'),
    PM2_5_avg=('PM2_5_avg', 'mean'),
    NO2_avg=('NO2_avg', 'mean'),
    CO_avg=('CO_avg', 'mean'),
    O3_avg=('O3_avg', 'mean')
).reset_index()



### Data Aggregation ###


## Air quality Data

AQ = pd.read_csv('/Users/silaskim/Desktop/THSS2/DataSets/Air Quality Data/Air_Quality_Data_Cleaned.csv',
                 delimiter=';',decimal='.',
                 names=['KOR_Name','Station_Code','M25_exc','PM10_exc','PM10_avg','PM25_avg','NO2_avg','CO_avg','O3_avg'],
                 header=0)


## Latitude and Longitude Data

LL = pd.read_csv('/Users/silaskim/Desktop/THSS2/DataSets/Socio-economic Data/Latitude_Longitude_Data.csv',
                 delimiter=',',decimal='.',names=['KOR_Name','latitude','longitude'],header=0)


## Population Data

POP = pd.read_csv('/Users/silaskim/Desktop/THSS2/DataSets/Socio-economic Data/Population_Data.csv',
                  delimiter=';',thousands='.',names=['KOR_Name','POP2018'],header=1)
POP['KOR_Name'] = POP['KOR_Name'].astype(str).str.strip()


## GRDP Data

GRDP = pd.read_csv('/Users/silaskim/Desktop/THSS2/DataSets/Socio-economic Data/Real_GRDP_Data.csv',
                   delimiter=';',names=['KOR_Name','GRDP2017','GRDP2018','GRDP2019','GRDP2020','GRDP2021'],header=0)

## Policy Data



## Aggrigate Data

dfs1 = [LL, AQ, POP, GRDP[['KOR_Name','GRDP2018']]]

from functools import reduce

DT1 = reduce(lambda left, right: pd.merge(left, right, on='KOR_Name', how='outer'), dfs1)



df = pd.read_csv('/Users/silaskim/Desktop/THSS2/DataSets/Data_Final.csv',delimiter = ';',decimal = ',')

# Define the dependent variables
exog_1= df[['Latitude','Longitude','City Size','Population',
            'Density','Education','GHG/capita','Energy','Water','Green','Waste','GRDP/capita',
            'ICLEI','PM10_2021','PM25_2021','NO2_2021']]

# Define the independent variables
endog = df[['2nd']]

# Add constant to the independent variables
exog = exog_1.copy()  # Create a copy of the DataFrame to avoid modifying the original
exog['const'] = 1  # Add a new column named 'const' with constant value 1


# Estimate parameters
probit_results = sm.Probit(endog,exog).fit()


# Print model summary
print(probit_results.summary())
