import sklearn
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import numpy as np
from module_a import polynom_3
from module_b import hyperbola

x = np.arange(1,100,1).reshape(1,-1)
y = polynom_3(x)
yy = hyperbola(x)

model1 = RandomForestRegressor()
model1.fit(x,y)

prediction1 = model1.predict(x)
model2 = RandomForestRegressor()
model2.fit(x,yy)

prediction2 = model2.predict(x)

print('MSE score of polynom_3:',mean_squared_error(y, prediction1))
print('MSE score of hyperbola:',mean_squared_error(yy, prediction2))

