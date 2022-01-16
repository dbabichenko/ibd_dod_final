from flask import Flask
from flask import request
from flask import render_template
import pandas as pd
import json

app = Flask(__name__)


@app.route('/')
def index():
    return "Welcome to ibd app"

// URL Example: http://localhost:5000/q/GENDER=='Male'/col/AGE/is_root/y
// 
@app.route('/q/<q>/col/<col>/is_root/<is_root>')
def query(q, col, is_root):
    df = pd.read_csv("ibd_source_data.csv")
    # sel_data_col = 'EMPLOYMENT_STATUS'
    # q = "GENDER=='Male' & MARITAL_STATUS == 'Married'"
    #sel_data_col = request.args.get('col')
    #q = request.args.get('q')
    #return df.query(q)[sel_data_col].value_counts().to_json()
    #return df.query(q)[col].value_counts().to_json()

    total_count = df.shape[0]

    children = []
    category_count = 0
    result = None
    if q != 'none':
        result = df.query(q)[col].value_counts()
    else:
        result = df[col].value_counts()
        
    category_count = int(str(result.sum()))
    


    for key, val in result.items():
        category_distr = (int(val) / int(category_count)) * 100
        overall_distr = (int(val) / int(total_count)) * 100
        temp = {
            'name' : key, 
            'count' : val, 
            'total_count': total_count, 
            'category_count': category_count, 
            'category_distr': category_distr,
            'overall_distr': overall_distr,
            'group': col,
            'children' : []
        }
        children.append(temp)
    
    if is_root == 'y':
        container = {'name' : col, 'children' : children}
        return json.dumps(container) #.to_json()
    else:
        return json.dumps(children) #.to_json()



@app.route('/<string:page_name>/')
def render_static(page_name):
        #return render_template('%s.html' % page_name)
        return render_template(page_name)
