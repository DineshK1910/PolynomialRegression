from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
file="/home/dinesh/codeit/c++/ML/FuelConsumptionCo2.csv"
df=pd.read_csv(file)
ndf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
plt.scatter(ndf.ENGINESIZE,ndf.CO2EMISSIONS)
plt.xlabel('engine size')
plt.ylabel('co2emissions')
plt.show()
msk=np.random.rand(len(df))<0.8
train=ndf[msk]
test=ndf[~msk]
trainx=np.asanyarray(train[['ENGINESIZE']])
trainy=np.asanyarray(train[['CO2EMISSIONS']])
testx=np.asanyarray(test[['ENGINESIZE']])
testy=np.asanyarray(test[['CO2EMISSIONS']])
poly=PolynomialFeatures(degree=3)
trainxp=poly.fit_transform (trainx)
print(trainxp)
clf=LinearRegression()
trainy_=clf.fit(trainxp,trainy)
testxp=poly.transform(testx)
testy_=clf.predict(testxp)
print(testy)
print("r2_score:%.2f"%r2_score(testy,testy_))