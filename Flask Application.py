#!/usr/bin/env python
# coding: utf-8

# In[10]:


from flask import Flask, request, render_template
import pickle

# Load the logistic regression model
with open('logistic_regression_model.pickle', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        tweet = request.form['tweet']
        prediction = model.predict([tweet])[0]
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




