import sys
# print("we are running ",sys.version)
from flask import Flask, render_template, flash, redirect, request, url_for, logging, jsonify
from wtforms import Form, TextField, FileField, validators
import predict
import data_helpers
import keras.models
import os
import json
import timeit
import requests as API_REQUESTS
import sys

sys.path.insert(0, 'Word2VecProto/scripts')
import word2vec

print(word2vec.train_prototype_model())

# ---------------------- Parameters section -------------------
#

startT = timeit.default_timer()
data_source = "pickle"  # keras_data_set|local_dir
print('Load Vocabulary')
vocabulary = predict.load_dict(data_source)
print('Load Model')
loaded_model = predict.load_model()
print('Load W2V')
predict.load_w2v()
stopT = timeit.default_timer()
print('Load duration: {}'.format(stopT - startT))

startT = timeit.default_timer()
print('Load Tiny Prototype')
tiny_mode = word2vec.train_prototype_model()
stopT = timeit.default_timer()
print('Load Duration for  Prototype{}'.format(stopT - startT))

#
# ---------------------- Parameters end -----------------------

app = Flask(__name__, static_url_path="/models", static_folder='./models')


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/qualify_notes_headline', methods=['POST'])
def qualify_notes_headline():
    problem = request.get_json()["headline"]
    note_type = request.get_json()["note_type"]
    try:
        if note_type != None:
            prediction=word2vec.evaluate_sentence(problem)
        else:
            prediction = predict.predict_Problem(loaded_model, vocabulary, problem)
            # print("Prediction Result",json.dumps(prediction, indent=4))
            prediction = prediction[0]
            prediction = prediction.tolist()[0][0]
            prediction = prediction * 100 if prediction else 0
        if prediction <= 40:
            bg_col = 'red'
        elif prediction > 40 and prediction < 60:
            bg_col = 'yellow'
        else:
            bg_col = 'green'

        return jsonify(quality=[bg_col, prediction])
    except:
        return jsonify(quality=["0", 0])


@app.route('/test1', methods=['GET', 'POST'])
def test1():
    server_error = True
    form = Test1Form(request.form)
    # print request.form
    problem = ""
    prediction = ""
    form_class = ""
    addSuccess = {}
    # try:
    if request.method == 'POST' and form.validate() and request.form['submit'] == "Predict":
        problem = form.problem.data
        prediction = predict.predict_Problem(loaded_model, vocabulary, problem)

        # print("PREDICTION TESTED",json.dumps(prediction, indent=4))
        prediction = prediction[0]
        prediction = prediction.tolist()[0][0]
        prediction = prediction * 100 if prediction else None
        if prediction <= 40:
            form_class = 'bg-danger'
        elif prediction > 40 and prediction < 60:
            form_class = 'bg-warning'
        else:
            form_class = 'bg-success'
        server_error = False

    if request.method == 'POST' and form.validate() and request.form['submit'] == "Add Problem":
        # Add Problem to rt-polarity.pog & a non problem to rt-polarity.neg
        success, errors = data_helpers.add_Problem(form.problem.data)
        if success:
            addSuccess['text'] = errors
            addSuccess['form_class'] = 'bg-success'
        else:
            addSuccess['text'] = errors
            addSuccess['form_class'] = 'bg-danger'
        server_error = False

    if request.method not in ['POST', 'GET'] and server_error:
        # print("Errors to handle in one of these entities:","problem",problem,"prediction",prediction,"form_class",form_class,"addSuccess",addSuccess,"form",form,sep="\n")
        errors = ["Ambiguos Input"]
        form_class = 'bg-danger'
        addSuccess['text'] = errors
        addSuccess['form_class'] = 'bg-danger'
    return render_template('test1.html', problem=problem, prediction=str(prediction), form_class=form_class,
                           addSuccess=addSuccess, form=form)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm(request.form)
    prediction = ""
    add_problems = []
    # print(request.files[form.problem.name])
    print(request)

    if request.method == 'POST':
        if request.form.getlist("add_problem"):
            cb_items = request.form.getlist("add_problem")
            for problem in cb_items:
                addSuccess = {}
                success, errors = data_helpers.add_Problem(problem)
                if success:
                    addSuccess = [problem, errors[0], 'bg-success']
                else:
                    addSuccess = [problem, errors[0], 'bg-danger']
                add_problems.append(addSuccess)

        if form.problem.name in request.files:
            sentence_text = request.files[form.problem.name].read()
            sentence_list = sentence_text.split('\n')
            form_class_tmp, prediction_tmp, a, b = predict.predict_Problem_List(loaded_model, vocabulary, sentence_list)
            prediction = (zip(sentence_list, form_class_tmp, prediction_tmp))

    return render_template('upload.html', form=form, prediction=prediction, add_problem=add_problems)


@app.route('/about')
def about():
    res = API_REQUESTS.get('http://localhost:5000/api/v1/expose_db_notes')
    response = res.json()
    return render_template('about.html', notes=response["canvas"])


"""
Text Generation Api
"""


@app.route('/suggest_description', methods=['POST'])
def suggest_description():
    req = request.get_json()
    return jsonify(response=predict.text_optimizer(req["canvas_field"]))


class Test1Form(Form):
    problem = TextField('Problem formulation', [
        validators.required(),
        validators.Length(min=10, max=1000)
    ])


class UploadForm(Form):
    problem = FileField('Upload a List of Problems', [
        validators.required("List of Problem is required")
    ])


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
