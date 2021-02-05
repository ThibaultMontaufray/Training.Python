import os
import mpld3
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import SeabornFig2Grid as sfg
import matplotlib.gridspec as gridspec

from http.server import HTTPServer, CGIHTTPRequestHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


# Make sure the server is created at current directory
os.chdir('.')

# Create server object listening the port 80
server_object = HTTPServer(server_address=('', 80), RequestHandlerClass=CGIHTTPRequestHandler)

########################
# using the models
########################

flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')

flower_example = {"sepal_length":5.1, "sepal_width":3.5, "petal_length":1.4, "petal_width":0.2 }

def return_prediction(model,scaler,sample_json):
    s_len = sample_json["sepal_length"]
    s_wid = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_wid = sample_json["petal_width"]

    flower = [[s_len,s_wid,p_len,p_wid]]
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    flower = scaler.transform(flower)

    class_ind = model.predict_classes(flower)
    return classes[class_ind]

print(return_prediction(flower_model, flower_scaler, flower_example))

########################
# Write figure to html
########################

fig = plt.figure(figsize = (8,8))
gs = gridspec.GridSpec(3, 1)
mg0 = sfg.SeabornFig2Grid(g0, fig, gs[0])
mg1 = sfg.SeabornFig2Grid(g1, fig, gs[1])
mg2 = sfg.SeabornFig2Grid(g2, fig, gs[2])
#mg3 = sfg.SeabornFig2Grid(g3, fig, gs[3])
gs.tight_layout(fig)

html_str = mpld3.fig_to_html(fig)
Html_file= open("html/graph.html","w")
Html_file.write(html_str)
Html_file.close()

# Start the web server
server_object.serve_forever()
