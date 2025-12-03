# import pickle
# from flask import Flask,request,jsonify,render_template
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# application = Flask(__name__)
# app=application

# ## import ridge regresor model and standard scaler pickle
# ridge_model=pickle.load(open('models/ridge.pkl','rb'))
# standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

# ## Route for home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predictdata',methods=['GET','POST'])
# def predict_datapoint():
#     if request.method=='POST':
#         Temperature=float(request.form.get('Temperature'))
#         RH = float(request.form.get('RH'))
#         Ws = float(request.form.get('Ws'))
#         Rain = float(request.form.get('Rain'))
#         FFMC = float(request.form.get('FFMC'))
#         DMC = float(request.form.get('DMC'))
#         ISI = float(request.form.get('ISI'))
#         Classes = float(request.form.get('Classes'))
#         Region = float(request.form.get('Region'))

#         new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
#         result=ridge_model.predict(new_data_scaled)

#         return render_template('home.html',result=result[0])

#     else:
#         return render_template('home.html')


# if __name__=="__main__":
#     app.run(host="0.0.0.0")


# import pickle
# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # WSGI application object – name MUST be "application"
# application = Flask(__name__)

# ## import ridge regressor model and standard scaler pickle
# ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
# standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# ## Route for home page
# @application.route('/')
# def index():
#     return render_template('index.html')

# @application.route('/predictdata', methods=['GET', 'POST'])
# def predict_datapoint():
#     if request.method == 'POST':
#         Temperature = float(request.form.get('Temperature'))
#         RH = float(request.form.get('RH'))
#         Ws = float(request.form.get('Ws'))
#         Rain = float(request.form.get('Rain'))
#         FFMC = float(request.form.get('FFMC'))
#         DMC = float(request.form.get('DMC'))
#         ISI = float(request.form.get('ISI'))
#         Classes = float(request.form.get('Classes'))
#         Region = float(request.form.get('Region'))

#         new_data_scaled = standard_scaler.transform(
#             [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
#         )
#         result = ridge_model.predict(new_data_scaled)

#         return render_template('home.html', result=result[0])
#     else:
#         return render_template('home.html')

# if __name__ == "__main__":
#     # Run the same WSGI object in local dev
#     application.run(host="0.0.0.0")


import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# WSGI callable – Elastic Beanstalk will use THIS
application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result=result[0])

    return render_template('home.html')

# Only for local run (not used by EB)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
