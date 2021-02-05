from flask import Flask
app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/")
def index():
    return '<h1>WELCOME TO BASIC API</h1>'

#if __name__=='main':
print(__name__)
app.run()
