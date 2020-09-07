from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    pass
    # return app.send_static_file('index.html')

@app.route('/api/image', methods=['POST'])
def scan():
    pass

@app.route('/api/board', methods=['POST'])
def solve():
    pass