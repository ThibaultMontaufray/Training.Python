import os
import mpld3
import numpy as np
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
#Create the Python figure
########################

#Set the size of the matplotlib canvas
df = pd.read_csv('data/fake_reg.csv')
g0 = sns.pairplot(df)
g0.map(sns.histplot)
g0.map_offdiag(sns.scatterplot)

X = df[['feature1', 'feature2']].values
y = df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(" -> create senquential")
model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

print(" -> compile")
model.compile(optimizer='rmsprop', loss='mse')

print(" -> fit")
model.fit(x=X_train, y=y_train,epochs=250)

loss_df = pd.DataFrame(model.history.history)
g1 = sns.relplot(data=loss_df)


print(" -> evaluate test : " + str(model.evaluate(X_test,y_test,verbose=0)))
print(" -> evaluate train : " + str(model.evaluate(X_train,y_train,verbose=0)))

test_predictions = model.predict(X_test)
test_predictions = pd.Series(test_predictions.reshape(300,))
pred_df = pd.DataFrame(y_test,columns=['Test True Y'])
pred_df = pd.concat([pred_df, test_predictions], axis = 1)
pred_df.columns = ['Test True Y', 'Predictions']
g2 = sns.relplot(x='Test True Y',y='Predictions',data=pred_df)

print(df.describe())

########################
# using the models
########################

new_gem = [[998,1000]]
new_gem = scaler.transform(new_gem)
print("test new gem")
print(new_gem)
print(model.predict(new_gem))

model.save('my_gem_model.h5')
later_model = load_model('my_gem_model.h5')
print(later_model.predict(new_gem))

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
