#step1
import pandas as pd
d=pd.read_csv('monkeypox-dataset.csv')

#step2
x=d.iloc[:,:-1].values
y=d.iloc[:,-1:].values

#step3
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)


#step4
from sklearn.preprocessing import StandardScaler
sst=StandardScaler()
X_train=sst.fit_transform(xtrain)   #Normalizing
X_test=sst.transform(xtest)

#step5
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

#step6
rfc.fit(xtrain,ytrain)

#step7
y_pred=rfc.predict(xtest)
print(y_pred)

print("Training Accuracy",rfc.score(xtrain,ytrain))
print("Testing Accuracy",rfc.score(xtest,ytest))
print("Overall Accuracy:",rfc.score(sst.transform(x),y))


#step8
#Pickling - DePickling
import pickle
pickle.dump(rfc,open('model.pkl','wb')) #we are serializing our model by creating model.pkl file where we are dumping rf - mode (trained)
print("Model is dumped")

