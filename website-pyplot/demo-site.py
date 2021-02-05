import os
import mpld3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from http.server import HTTPServer, CGIHTTPRequestHandler

# Make sure the server is created at current directory
os.chdir('.')

# Create server object listening the port 80
server_object = HTTPServer(server_address=('', 80), RequestHandlerClass=CGIHTTPRequestHandler)

########################
#Create the Python figure
########################

#Set the size of the matplotlib canvas
fig = plt.figure(figsize = (8,8))

#Create subplots in Python

########################
#Subplot 1
########################
sub1 = plt.subplot(3,2,1)

plt.xlim(0,4)
plt.ylim(0,300)

plt.title('test lineaire')
x = [0 ,1, 2, 3, 4]
y = [100, 200, 300, 400, 500]
plt.plot(x, y, color="#7F0000", marker='o', markersize=15, linestyle='-')

########################
#Subplot 2
########################
sub2 = plt.subplot(3,2,2)

#Add titles to the chart and axes
plt.title('test scatter')
plt.xlabel('Room')
plt.ylabel('Price')

housing = pd.DataFrame({"Room":[1,1,2,2,2,3,3,3],"Price":[100,120,190,200,230,310,330,305]})
plt.scatter(housing["Room"],housing["Price"])

########################
#Subplot 3
########################
sub3 = plt.subplot(3,2,3)

#Add titles to the chart and axes
plt.title('test distplot')

df = pd.read_csv("data/heart.csv")
sns.distplot(df['age'],kde=True,bins=50,color="#7F0000")

########################
#Subplot 4
########################
sub4 = plt.subplot(3,2,4)

#Add titles to the chart and axes
plt.title('test countplot')

df = pd.read_csv("data/heart.csv")
sns.countplot(x='cp',data=df,hue='sex',palette='terrain')

########################
#Subplot 5
########################
sub4 = plt.subplot(3,2,5)

#Add titles to the chart and axes
plt.title('test boxplot')

df = pd.read_csv("data/heart.csv")
sns.boxplot(x='target', y='thalach', data=df,hue='sex')

########################
#Subplot 6
########################
sub4 = plt.subplot(3,2,6)

#Add titles to the chart and axes
plt.title('test scatterplot')

df = pd.read_csv("data/heart.csv")
sns.scatterplot(x='chol', y='trestbps', data=df,hue='sex', palette='Dark2', size='age')

########################
# Write figure to html
########################

fig.tight_layout()

html_str = mpld3.fig_to_html(fig)
Html_file= open("html/graph.html","w")
Html_file.write(html_str)
Html_file.close()

# Start the web server
server_object.serve_forever()
