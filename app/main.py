import pickle
import numpy as np

from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from app.prediction import Prediction
from flask_cors import CORS, cross_origin

app = Flask(__name__)
model = pickle.load(open('model/front_end_model.pkl', 'rb'))
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
prediction = Prediction()

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    age = request.form['age(y)']
    career = request.form['career']
    exp = request.form['exp(y)']
    language = request.form.getlist('language')
    web_pro_lang = request.form.getlist('web_pro_lang')
    UI_Libs = request.form['UI_Libs']
    front_dev = request.form['front_dev']
    learning_src = request.form.getlist('learning_src')
    duration = request.form['duration(m)']
    factor_market_needs = request.form['factor_market_needs']
    factor_compensation = request.form['factor_compensation']
    website_function_present = request.form['website_function_present']
    website_content = request.form['website_content']
    website_graphics = request.form['website_graphics']
    website_functions = request.form['website_functions']

    feat_lang = prediction.transform_input(language,2)
    feat_web_pro_lang = prediction.transform_input(web_pro_lang,4)
    feat_learning_src = prediction.transform_input(learning_src,4)

    tmp_str = str(age) + str(career) + str(exp) + \
        str(feat_lang) + str(feat_web_pro_lang) + str(UI_Libs) + \
        str(front_dev) + str(feat_learning_src)+ \
        str(duration) + str(factor_market_needs) + \
        str(factor_compensation) + str(website_function_present) + \
        str(website_content) + str(website_graphics) + \
        str(website_functions) 

    #Check if input is empty or not
    #Check if enough features or not
    if not tmp_str or len(tmp_str) != 41:
        return jsonify(result="required more information",status="faile")

    else:
        tmp_str_split = list(tmp_str)

        tmp_int = []
        for i in tmp_str_split:
            tmp_int.append(int(i))

        final_features = np.array(tmp_int)
        final_input = final_features.reshape(1,-1)
        results = prediction.predict(final_input)
        output = results[0]

        return jsonify(result=output,status="success")

