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
import cPickle 

df_test = pandas.read_csv("test.csv")
test_smiles = df_test.smiles
X_test =  list(test_smiles.apply(lambda x: list(map(int, rdmolops.LayeredFingerprint(Chem.MolFromSmiles(x),fpSize=1024).ToBitString()))))
del df_train, train_smiles
print "Test data made"
testfile = open("testdata", 'wb')
cPickle.dump(X_test, testfile)
testfile.close()
print "test data stored"

