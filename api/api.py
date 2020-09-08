from flask import Flask, request, Response, jsonify
import solver

app = Flask(__name__)

@app.route('/')
def index():
    pass
    # return app.send_static_file('index.html')

@app.route('/api/image', methods=['POST'])
def scan():
    pass

@app.route('/api/puzzle', methods=['POST'])
def solve():
    if request.content_type == 'application/json' :
        try:
            data = request.get_json()
            grid = data["puzzle"]
            if len(grid) != 81:
                raise Exception("")
            board = []
            for i in range(0,81,9):
                board.append(grid[i:i+9])
            for i in range(9):
                print(board[i])
            del grid[:]
            if not solver.backTrack(board):
                return jsonify(data)
            for i in range(9):
                grid.extend(board[i])
            return jsonify(data)
        except:
            return Response("Invalid JSON", status=401, mimetype='text/plain')
    else:
        return Response("Invalid content type", status=401, mimetype='text/plain')