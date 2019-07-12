#IMPORT PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#IMPORTING .CSV FILE
dataset = pd.read_csv('position_salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(X,y)

y_predict = regressor.predict(np.array([8.5]).reshape(-1,1))

# VISUALIZATION
X_grid = np.arange(min(X),max(X),0.01)

X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color = 'red')

plt.plot(X_grid,regressor.predict(X_grid),color  = 'blue')

plt.title("TRUTH OR BLUFF")

plt.xlabel("Position of level")

plt.ylabel("Salay")

plt.savefig("decision tree.png")
