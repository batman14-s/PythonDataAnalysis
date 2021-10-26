"""
NAME-SWEETY AGARWAL
B20234
PH-8107876050
"""
def space():
    for i in range(5):
        print()

import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('landslide_data3_miss.csv')

print(df.shape)
print("THERE ARE TOTAL 945 ROWS AND 9 COLUMNS")

# 1)
print("1)")

# FINDING NUMBER OF NANs PER COLUMN...
NANs = df.isna().sum()
print(type(NANs))

# CONVERTING TYPE SERIES INTO DATAFRAME AND NAMING COLUMNS
NANs = NANs.to_frame().reset_index()
NANs.columns = ['ATTRIBUTES', 'COUNT OF NANs']
print(NANs)

# creating the bar plot
plt.style.use('fivethirtyeight')
plt.bar(NANs['ATTRIBUTES'], NANs['COUNT OF NANs'], color='maroon',
        width=0.4)

plt.xlabel("ATTRIBUTES UNDER CSV FILE")
plt.ylabel("COUNT OF NANs")
plt.title("BAR GRAPH FOR ANALYZING NUMBER OF NANs IN CSV FILE")
plt.tight_layout()
plt.show()

space()

# 2a)
print("2a)")
'''Deleting (drop) the tuples (rows) having missing valies in target attribute.'''

df = df.dropna(how='any',
               subset=['stationid'])
print("Modified Dataframe : ")

print(df.shape)
# modified df has 926 rows
print('number of tuples deleted =', (945 - 926))

space()


# 2b)
print("2b)")
'''
Total columns=9
therefore, more than 1/3rd implies, more than 3 values as MISSING VALUES.
'''

df = df.dropna(axis=0, thresh=7)

print(df.shape)
print('number of tuples deleted  =', (945 - 891))



space()

# 3)
print("3)")
'''
Finding the number of missing values in each attribute after following steps above.
'''
NA2 = df.isna().sum()


'''
Finding total number of NAN values in modified dataframe.
'''
print("NAN VALUES PER COLUMN")
print(df.isna().sum())  # NAN values per column
total_na = df.isna().sum().sum()
print('Total NANs in this dataframe are', total_na)



space()
# 4)
print("4a)")
'''
Replacing the missing values by mean of their respective attribute. 
'''

column_means = df.mean(numeric_only=True)
df_a = df.fillna(column_means)

'''Compute the mean, median, mode and standard deviation for each attributes
and compare the same with that of the original file.'''
df2 = pd.read_csv('landslide_data3_original.csv')

print('Mean,median, mode, and stdev of previous file after replacing missing values by mean')
print(df_a.mean(numeric_only=True))
print(df_a.median(numeric_only=True))
print(df_a.mode(numeric_only=True))
print(df_a.std(numeric_only=True))

print('Mean,median, mode, and stdev of original file')
print(df2.mean(numeric_only=True))
print(df2.median(numeric_only=True))
print(df2.mode(numeric_only=True))
print(df2.std(numeric_only=True))

'''Calculate the root mean square error (RMSE) between the original and
replaced values for each attributes. (Get original values from original file
provided). Compute RMSE using the equation (1). Plot these RMSE with
respect to the attributes. 
'''
########################
T1 = df_a['temperature']
T2 = df2['temperature']

diff_T = T1.subtract(T2)

diff_T = diff_T.to_frame().reset_index()

diff_T.rename(columns={'temperature': 'difference of temperatures'},
              inplace=True)

diff_T = diff_T.dropna(how='any',
                       subset=['difference of temperatures', 'index'])

diff_T['square_Temp'] = diff_T['difference of temperatures'] ** 2

Total_diffT = diff_T['square_Temp'].sum()

print("RMSE OF TEMPERATURE=", math.sqrt(Total_diffT / 34))

##################
H1 = df_a['humidity']
H2 = df2['humidity']

diff_H = H1.subtract(H2)
diff_H = diff_H.to_frame().reset_index()

diff_H.rename(columns={'humidity': 'difference of humidity'},
              inplace=True)

diff_H = diff_H.dropna(how='any',
                       subset=['difference of humidity', 'index'])

diff_H['square_humidity'] = diff_H['difference of humidity'] ** 2

Total_diffH = diff_H['square_humidity'].sum()

print("RMSE OF HUMIDITY =", math.sqrt(Total_diffH / 13))

##################
P1 = df_a['pressure']
P2 = df2['pressure']

diff_P = P1.subtract(P2)
diff_P = diff_P.to_frame().reset_index()

diff_P.rename(columns={'pressure': 'difference of pressure'},
              inplace=True)

diff_P = diff_P.dropna(how='any',
                       subset=['difference of pressure', 'index'])

diff_P['square_pressure'] = diff_P['difference of pressure'] ** 2

Total_diffP = diff_P['square_pressure'].sum()

print("RMSE OF PRESSURE", math.sqrt(Total_diffP / 41))

##################
R1 = df_a['rain']
R2 = df2['rain']

diff_R = R1.subtract(R2)
diff_R = diff_R.to_frame().reset_index()

diff_R.rename(columns={'rain': 'difference of rain'},
              inplace=True)

diff_R = diff_R.dropna(how='any',
                       subset=['difference of rain', 'index'])

diff_R['square_rain'] = diff_R['difference of rain'] ** 2

Total_diffR = diff_R['square_rain'].sum()

print("RMSE OF RAIN =", math.sqrt(Total_diffR / 6))
##################
# lightavgw/o0
L1 = df_a['lightavgw/o0']
L2 = df2['lightavgw/o0']

diff_L = L1.subtract(L2)
diff_L = diff_L.to_frame().reset_index()

diff_L.rename(columns={'lightavgw/o0': 'difference of light'},
              inplace=True)

diff_L = diff_L.dropna(how='any',
                       subset=['difference of light', 'index'])

diff_L['square_light'] = diff_L['difference of light'] ** 2

Total_diffL = diff_L['square_light'].sum()

print("RMSE OF LIGHTAVGW/O0= ", math.sqrt(Total_diffL / 15))

##################
# lightmax
LX1 = df_a['lightmax']
LX2 = df2['lightmax']

diff_LX = LX1.subtract(LX2)
diff_LX = diff_LX.to_frame().reset_index()

diff_LX.rename(columns={'lightmax': 'difference of lux'},
               inplace=True)

diff_LX = diff_LX.dropna(how='any',
                         subset=['difference of lux', 'index'])

diff_LX['square_lux'] = diff_LX['difference of lux'] ** 2

Total_diffLX = diff_LX['square_lux'].sum()

print("RMSE OF LIGHTMAX= ", math.sqrt(Total_diffLX / 1))
##################

M1 = df_a['moisture']
M2 = df2['moisture']

diff_M = M1.subtract(M2)
diff_M = diff_M.to_frame().reset_index()

diff_M.rename(columns={'moisture': 'difference of moisture'},
              inplace=True)

diff_M = diff_M.dropna(how='any',
                       subset=['difference of moisture', 'index'])

diff_M['square_moisture'] = diff_M['difference of moisture'] ** 2

Total_diffM = diff_M['square_moisture'].sum()

print("RMSE OF MOISTURE= ", math.sqrt(Total_diffM / 6))

# creating the dataset
data = {'Temperature': 3.65, 'Humidity': 6.95, 'Pressure': 21.05,
        'Rain': 10539.42, 'Lightavgw': 2055.50, 'Lightmax': 9424.77, 'Moisture': 37.01}
attributes = list(data.keys())
RMSE = list(data.values())

fig2 = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(attributes, RMSE, color='yellow',
        width=0.4, log=True)

plt.xlabel("Attribute")
plt.ylabel("RMSE")
plt.title("BAR GRAPH INDICATING RMSE OF EACH ATTRIBUTE WHEN NAN's \nREPLACED BY MEAN OF EACH ATTRIBUTE")
plt.tight_layout()
plt.show()
space()
##########################################################################################

print("4b")
'''
Replacing the missing values by interpolation of their respective attribute. 
'''

column_interpolation = df.interpolate(numeric_only=True)
df_b = df.fillna(column_interpolation)

'''Compute the mean, median, mode and standard deviation for each attributes
and compare the same with that of the original file.'''
df2 = pd.read_csv('landslide_data3_original.csv')

print('Mean, mode, median and standard deviation before and after replacing missing values by linear interpolation technique ')
print(df_b.mean(numeric_only=True))
print(df_b.median(numeric_only=True))
print(df_b.mode(numeric_only=True))
print(df_b.std(numeric_only=True))

print('Mean,median, mode, and stdev of original file')
print(df2.mean(numeric_only=True))
print(df2.median(numeric_only=True))
print(df2.mode(numeric_only=True))
print(df2.std(numeric_only=True))

'''Calculate the root mean square error (RMSE) between the original and
replaced values for each attributes. (Get original values from original file
provided). Compute RMSE using the equation (1). Plot these RMSE with
respect to the attributes. 
'''
########################
T1 = df_b['temperature']
T2 = df2['temperature']

diff_T = T1.subtract(T2)

diff_T = diff_T.to_frame().reset_index()

diff_T.rename(columns={'temperature': 'difference of temperatures'},
              inplace=True)

diff_T = diff_T.dropna(how='any',
                       subset=['difference of temperatures', 'index'])

diff_T['square_Temp'] = diff_T['difference of temperatures'] ** 2

Total_diffT = diff_T['square_Temp'].sum()

print("RMSE OF TEMPERATURE=", math.sqrt(Total_diffT / 34))

##################
H1 = df_b['humidity']
H2 = df2['humidity']

diff_H = H1.subtract(H2)
diff_H = diff_H.to_frame().reset_index()

diff_H.rename(columns={'humidity': 'difference of humidity'},
              inplace=True)

diff_H = diff_H.dropna(how='any',
                       subset=['difference of humidity', 'index'])

diff_H['square_humidity'] = diff_H['difference of humidity'] ** 2

Total_diffH = diff_H['square_humidity'].sum()

print("RMSE OF HUMIDITY =", math.sqrt(Total_diffH / 13))

##################
P1 = df_b['pressure']
P2 = df2['pressure']

diff_P = P1.subtract(P2)
diff_P = diff_P.to_frame().reset_index()

diff_P.rename(columns={'pressure': 'difference of pressure'},
              inplace=True)

diff_P = diff_P.dropna(how='any',
                       subset=['difference of pressure', 'index'])

diff_P['square_pressure'] = diff_P['difference of pressure'] ** 2

Total_diffP = diff_P['square_pressure'].sum()

print("RMSE OF PRESSURE", math.sqrt(Total_diffP / 41))

##################
R1 = df_b['rain']
R2 = df2['rain']

diff_R = R1.subtract(R2)
diff_R = diff_R.to_frame().reset_index()

diff_R.rename(columns={'rain': 'difference of rain'},
              inplace=True)

diff_R = diff_R.dropna(how='any',
                       subset=['difference of rain', 'index'])

diff_R['square_rain'] = diff_R['difference of rain'] ** 2

Total_diffR = diff_R['square_rain'].sum()

print("RMSE OF RAIN =", math.sqrt(Total_diffR / 6))
##################
# lightavgw/o0
L1 = df_b['lightavgw/o0']
L2 = df2['lightavgw/o0']

diff_L = L1.subtract(L2)
diff_L = diff_L.to_frame().reset_index()

diff_L.rename(columns={'lightavgw/o0': 'difference of light'},
              inplace=True)

diff_L = diff_L.dropna(how='any',
                       subset=['difference of light', 'index'])

diff_L['square_light'] = diff_L['difference of light'] ** 2

Total_diffL = diff_L['square_light'].sum()

print("RMSE OF LIGHTAVGW/O0= ", math.sqrt(Total_diffL / 15))

##################
# lightmax
LX1 = df_b['lightmax']
LX2 = df2['lightmax']

diff_LX = LX1.subtract(LX2)
diff_LX = diff_LX.to_frame().reset_index()

diff_LX.rename(columns={'lightmax': 'difference of lux'},
               inplace=True)

diff_LX = diff_LX.dropna(how='any',
                         subset=['difference of lux', 'index'])

diff_LX['square_lux'] = diff_LX['difference of lux'] ** 2

Total_diffLX = diff_LX['square_lux'].sum()

print("RMSE OF LIGHTMAX= ", math.sqrt(Total_diffLX / 1))
##################

M1 = df_b['moisture']
M2 = df2['moisture']

diff_M = M1.subtract(M2)
diff_M = diff_M.to_frame().reset_index()

diff_M.rename(columns={'moisture': 'difference of moisture'},
              inplace=True)

diff_M = diff_M.dropna(how='any',
                       subset=['difference of moisture', 'index'])

diff_M['square_moisture'] = diff_M['difference of moisture'] ** 2

Total_diffM = diff_M['square_moisture'].sum()

print("RMSE OF MOISTURE= ", math.sqrt(Total_diffM / 6))

# creating the dataset
data = {'Temperature': 1.32, 'Humidity': 6.32, 'Pressure': 6.18,
        'Rain': 233.41, 'Lightavgw': 7328.81, 'Lightmax': 0, 'Moisture': 15.42}
attributes = list(data.keys())
RMSE = list(data.values())

# creating the bar plot
plt.bar(attributes, RMSE, color='green',
        width=0.4, log=True)

plt.xlabel("Attribute")
plt.ylabel("RMSE")
plt.title("BAR GRAPH INDICATING RMSE OF EACH ATTRIBUTE WHEN NAN's \nREPLACED WITH INTERPOLATION TECHNIQUE")
plt.tight_layout()
plt.show()
space()
###############################################################
#5
print("5a and 5b")
"""After replacing the missing values by interpolation method, find and list the outliers
in the attributes “temperature” and “rain"""
import seaborn as sns

sns.set(style='whitegrid')

g = sns.boxplot(data=df_b[['temperature']])
plt.title('BOXPLOT OF TEMPERATURE')
plt.tight_layout()
plt.show()
print(df_b.temperature.quantile([0.25, 0.5, 0.75]))

Q1 = 18.003495
Q2 = 22.139860
Q3 = 24.411911
IQR = Q3 - Q1
low_lim = Q1 - 1.5 * IQR
up_lim = Q3 + 1.5 * IQR
print('low_limit is', low_lim)
print('up_limit is', up_lim)
outlier = []
for x in df_b['temperature']:
    if (x > up_lim) or (x < low_lim):
        outlier.append(x)
print(' outlier in the dataset is', outlier)
median_temp = df_b['temperature'].median()
print("median of temperature attribute is", median_temp)
df_b.loc[df_b.temperature < 8.39, 'temperature'] = np.nan
df_b.fillna(median_temp, inplace=True)

sns.set(style='whitegrid')

g = sns.boxplot(data=df_b[['temperature']])
plt.title('BOXPLOT OF TEMPERATURE AFTER REPLACING \n OUTLIERS WITH MEDIAN.')
plt.tight_layout()
plt.show()
###########
sns.set(style='whitegrid')

g = sns.boxplot(data=df_b[['rain']])
plt.title('BOXPLOT OF RAIN')
plt.tight_layout()
plt.show()
print(df_b.rain.quantile([0.25, 0.5, 0.75]))

Q1 = 0.0
Q2 = 15.75
Q3 = 1041.75
IQR = Q3 - Q1
low_lim = Q1 - 1.5 * IQR
up_lim = Q3 + 1.5 * IQR
print('low_limit is', low_lim)
print('up_limit is', up_lim)
outlier = []
for x in df_b['rain']:
    if (x > up_lim) or (x < low_lim):
        outlier.append(x)
print(' outlier in the dataset is', outlier)
print(len(outlier))
median_rain = df_b['rain'].median()
print("median of rain attribute is ", median_rain)
df_b.loc[df_b.rain > 2604.375, 'rain'] = np.nan
df_b.fillna(median_rain, inplace=True)

sns.set(style='whitegrid')
g = sns.boxplot(data=df_b[['rain']])
plt.title('BOXPLOT OF RAIN AFTER REPLACING OUTLIERS WITH MEDIAN.')
plt.show()
