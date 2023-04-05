from flask import Flask,render_template,request
from sklearn.preprocessing import StandardScaler
import pickle
#model.pkl -trained ml model

#Desirilze-read the binary file-ML model
clf=pickle.load(open('model.pkl','rb'))

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


app=Flask(__name__)

@app.route('/')  #Annnotation triggers the methods following-defalut annotation that randers the 1st web page to the browser
def hello():
    return render_template('indexs.html')

#jinja2-template engine-which would be going to template folder and selecting the webpage-hence folder name should be template




@app.route('/predict',methods=['post','Get'])
def predict_class():
    print([x for x in request.form.values()])
    features=[int(x) for x in request.form.values()]
    print(features)
    sst=StandardScaler().fit(xtrain)


    output=clf.predict(sst.transform([features]))
    print(output)

    if output[0]==0:
        return render_template('indexs.html',pred=f'He  has monkey pox disease')
    else:
        return render_template('indexs.html',pred=f'He didnt has monkey pox disease')

if __name__ =="__main__":
    app.run(debug=True)
