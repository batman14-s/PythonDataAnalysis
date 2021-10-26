import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('pima-indians-diabetes.csv')

# Printig the data
print(df)

# Finding mean of the attributes and printing Them

print("Mean of all the attributes are as follows:")
print("MEAN OF PREGS=", df['pregs'].mean())
print("MEAN OF PLAS=", df['plas'].mean())
print("MEAN OF PRES=", df['pres'].mean())
print("MEAN OF SKIN=", df['skin'].mean())
print("MEAN OF TEST=", df['test'].mean())
print("MEAN OF BMI=", df['BMI'].mean())
print("MEAN OF PEDI=", df['pedi'].mean())
print("MEAN OF AGE=", df['Age'].mean())
print()
print()
# Finding median of the attributes and printing Them

print("Median of all the attributes are as follows:")
print("MEDIAN OF PREGS=", df['pregs'].median())
print("MEDIAN OF PLAS=", df['plas'].median())
print("MEDIAN OF PRES=", df['pres'].median())
print("MEDIAN OF SKIN=", df['skin'].median())
print("MEDIAN OF TEST=", df['test'].median())
print("MEDIAN OF BMI=", df['BMI'].median())
print("MEDIAN OF PEDI=", df['pedi'].median())
print("MEDIAN OF AGE=", df['Age'].median())
print()
print()

# Finding mode of entire dataframe together.

print("Mode of all the attributes are as follows:")
print("Mode OF PREGS=", df['pregs'].mode().iat[0])
print("Mode OF PLAS=", df['plas'].mode().iat[0])
print("Mode OF PRES=", df['pres'].mode().iat[0])
print("Mode OF SKIN=", df['skin'].mode().iat[0])
print("Mode OF TEST=", df['test'].mode().iat[0])
print("Mode OF BMI=", df['BMI'].mode().iat[0])
print("Mode OF PEDI=", df['pedi'].mode().iat[0])
print("Mode OF AGE=", df['Age'].mode().iat[0])
print()
print()

# Finding max of entire dataframe together.

print("Max of all the attributes are as follows:")
print("Max OF PREGS=", df['pregs'].max())
print("Max OF PLAS=", df['plas'].max())
print("Max OF PRES=", df['pres'].max())
print("Max OF SKIN=", df['skin'].max())
print("Max OF TEST=", df['test'].max())
print("Max OF BMI=", df['BMI'].max())
print("Max OF PEDI=", df['pedi'].max())
print("Max OF AGE=", df['Age'].max())
print()
print()

# Finding min of entire dataframe together.

print("Min of all the attributes are as follows:")
print("Min OF PREGS=", df['pregs'].min())
print("Min OF PLAS=", df['plas'].min())
print("Min OF PRES=", df['pres'].min())
print("Min OF SKIN=", df['skin'].min())
print("Min OF TEST=", df['test'].min())
print("Min OF BMI=", df['BMI'].min())
print("Min OF PEDI=", df['pedi'].min())
print("Min OF AGE=", df['Age'].min())
print()
print()

# Finding stdev of entire dataframe together.

print("STANDARD DEVIATION of all the attributes are as follows:")
print("Standard Deviation OF PREGS=", df['pregs'].std())
print("Standard Deviation OF PLAS=", df['plas'].std())
print("Standard Deviation OF PRES=", df['pres'].std())
print("Standard Deviation OF SKIN=", df['skin'].std())
print("Standard Deviation OF TEST=", df['test'].std())
print("Standard Deviation OF BMI=", df['BMI'].std())
print("Standard Deviation OF PEDI=", df['pedi'].std())
print("Standard Deviation OF AGE=", df['Age'].std())
print()
print()

#
plt.style.use('seaborn')
pregs_col = df['pregs']
plas_col = df['plas']
pres_col = df['pres']
skin_col = df['skin']
test_col = df['test']
bmi_col = df['BMI']
pedi_col = df['pedi']
age_col = df['Age']

# Scatter plot between ‘Age’ and each of the other attributes
fig, ax = plt.subplots(nrows=3, ncols=3)
ax[0, 0].scatter(pregs_col, age_col, label='pregs')
ax[0, 0].set_ylabel('Age')
ax[0, 0].set_xlabel('PREGS')
ax[0, 1].scatter(plas_col, age_col, label='plas')
ax[0, 1].set_xlabel('Age')
ax[0, 1].set_xlabel('PLAS')
ax[0, 2].scatter(pres_col, age_col, label='ples')
ax[0, 2].set_ylabel('Age')
ax[0, 2].set_xlabel('PLES')
ax[1, 0].scatter(skin_col, age_col, label='skin')
ax[1, 0].set_ylabel('Age')
ax[1, 0].set_xlabel('SKIN')
ax[1, 1].scatter(test_col, age_col, label='test')
ax[1, 1].set_ylabel('Age')
ax[1, 1].set_xlabel('TEST')
ax[1, 2].scatter(bmi_col, age_col, label='BMI')
ax[1, 2].set_ylabel('Age')
ax[1, 2].set_xlabel('BMI')
ax[2, 0].scatter(pedi_col, age_col, label='pedi')
ax[2, 0].set_ylabel('Age')
ax[2, 0].set_xlabel('PEDI')

plt.tight_layout()

plt.plot()
plt.show()

# Scatter plot between ‘BMI’ and each of the other attributes
fig, ax = plt.subplots(nrows=3, ncols=3)
ax[0, 0].scatter(pregs_col, bmi_col, label='pregs')
ax[0, 0].set_ylabel('BMI')
ax[0, 0].set_xlabel('PREGS')
ax[0, 1].scatter(plas_col, bmi_col, label='plas')
ax[0, 1].set_xlabel('BMI')
ax[0, 1].set_xlabel('PLAS')
ax[0, 2].scatter(pres_col, bmi_col, label='pres')
ax[0, 2].set_ylabel('BMI')
ax[0, 2].set_xlabel('PLES')
ax[1, 0].scatter(skin_col, bmi_col, label='skin')
ax[1, 0].set_ylabel('BMI')
ax[1, 0].set_xlabel('SKIN')
ax[1, 1].scatter(test_col, bmi_col, label='test')
ax[1, 1].set_ylabel('BMI')
ax[1, 1].set_xlabel('TEST')
ax[1, 2].scatter(age_col, bmi_col, label='BMI')
ax[1, 2].set_ylabel('BMI')
ax[1, 2].set_xlabel('AGE')
ax[2, 0].scatter(pedi_col, bmi_col, label='pedi')
ax[2, 0].set_ylabel('BMI')
ax[2, 0].set_xlabel('PEDI')

plt.tight_layout()

plt.plot()
plt.show()
print()
print()
print()


#FINDING CORRELATION COEFFICIENT OF AGE WITH OTHER ATTRIBUTES.
print('CORRELATUON COEFFICIENT OF AGE WITH OTHER ATTRIBUTES.')
print("CORRELATION COEFF B/W AGE AND PREGS ",age_col.corr(pregs_col))
print("CORRELATION COEFF B/W AGE AND PLAS ",age_col.corr(plas_col))
print("CORRELATION COEFF B/W AGE AND PRES ",age_col.corr(pres_col))
print("CORRELATION COEFF B/W AGE AND SKIN ",age_col.corr(skin_col))
print("CORRELATION COEFF B/W AGE AND TEST ",age_col.corr(test_col))
print("CORRELATION COEFF B/W AGE AND BMI ",age_col.corr(bmi_col))
print("CORRELATION COEFF B/W AGE AND PEDI ",age_col.corr(pedi_col))
print()
print()
print()


#FINDING CORRELATION COEFFICIENT OF BMI WITH OTHER ATTRIBUTES.
print('CORRELATUON COEFFICIENT OF AGE WITH OTHER ATTRIBUTES.')
print("CORRELATION COEFF B/W BMI AND PREGS ",bmi_col.corr(pregs_col))
print("CORRELATION COEFF B/W BMI AND PLAS ",bmi_col.corr(plas_col))
print("CORRELATION COEFF B/W BMI AND PRES ",bmi_col.corr(pres_col))
print("CORRELATION COEFF B/W BMI AND SKIN ",bmi_col.corr(skin_col))
print("CORRELATION COEFF B/W BMI AND TEST ",bmi_col.corr(test_col))
print("CORRELATION COEFF B/W BMI AND Age ",bmi_col.corr(age_col))
print("CORRELATION COEFF B/W BMI AND PEDI ",bmi_col.corr(pedi_col))
print()
print()
print()

#HISTOGRAM FOR PREGS ATTRIBUTE
data1 = df['pregs']
plt.hist(data1,bins=15)
plt.ylabel('Y-Axis')
plt.xlabel('X-axis')
plt.title('HISTOGRAM FOR PREGS')

plt.show()

print()
print()

#HISTOGRAM FOR SKIN ATTRIBUTE
data2 = df['skin']
plt.hist(data2, bins=30)
plt.ylabel('Y-Axis')
plt.xlabel('X-axis')
plt.title('Histogram for skin')

plt.show()

print()
print()


print()
print()
#Plotting the histogram of attribute ‘preg’ for each of the 2 classes individually

gk = df.groupby("class")
c1 = gk.get_group(1)
c0 = gk.get_group(0)
pregsclass1 = list(c1["pregs"])
pregsclass2 = list(c0["pregs"])
#plotting the histogram
plt.hist(pregsclass1,edgecolor= 'black',color= 'yellow')
plt.xlabel("Values")
plt.ylabel("Pregs of class1")
plt.title("Histogram For Pregs with class 1")
plt.show()
plt.hist(pregsclass2,edgecolor= 'red',color= 'black')
plt.xlabel("Values")
plt.ylabel("Pregs of class2")
plt.title("Histogram For Pregs with class 0")
plt.show()



# BOXPLOT

sns.set(style='whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))
g=sns.boxplot(data=df[['pregs','plas','pres','skin','test','BMI','pedi','Age']])
plt.title('BOXPLOTS OF ALL ATTRIBUTES')
plt.show()