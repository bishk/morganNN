import numpy 
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import pickle 

Xfile = open("X",'r') 
Yfile = open("Y",'r') 

X = pickle.load(Xfile)
Y = pickle.load(Yfile)

X.close()
Y.close()

print "Loading data done"

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(1024, input_dim=1024, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

seed = 7
#numpy.random.seed(seed)
# evaluate model with standardized dataset
#estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=1000, verbose=0)

#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv=kfold)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))



# # evaluate model with standardized dataset

# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=1000, verbose=1)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(1024, input_dim=1024, kernel_initializer='normal', activation='relu'))
	model.add(Dense(512, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=100, batch_size=1000, verbose=1)))
pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(pipeline, X, Y, cv=kfold)
#print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
pipeline.fit(X, Y)

Pfile = open("model", "wb")
pickle.dump(pipeline, Pfile)
Pfile.close()
