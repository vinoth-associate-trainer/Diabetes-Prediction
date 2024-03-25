from flask import Flask ,render_template ,request
import pickle
import sklearn
import numpy as np
from joblib import dump, load
app = Flask(__name__)

filename = './diabetes.joblib'

classifier = load(open(filename, 'rb'))


@app.route('/', methods=['POST','GET'])
def dia():
    if request.method == 'POST':
        a = request.form
        b=list(a)
        c=[]
        for i in b:
            d=request.form[i]
            d=float(d)
            c.append(d)
        a=np.array([c])
        my_prediction = classifier.predict(a)
        a=my_prediction
        if a==1:
            a='You Have diabetes problem'
        else:
            a='You didnt have diabetes issue'

        return render_template('diabetes.html', name=a)
    else:
        return render_template('diabetes.html')


if "__main__" == __name__:
    app.run()