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

@app.route('/predict',methods=['POST'])
def predict():  
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = xgmodel.predict(final_input)[0]
    return render_template('index.html', prediction_text='The predicted value is {}'.format(output)) 


if __name__=="__main__":
    app.run(debug=True)