import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd




app=Flask(__name__)
## Load the model
xgmodel=pickle.load(open('XGMODEL.pkl','rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = xgmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)