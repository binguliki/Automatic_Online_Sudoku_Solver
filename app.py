from flask import Flask , jsonify , request
from sudoku import ScanSudoku , Solve
import numpy as np
import cv2 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

@app.route("/sudoku_api" , methods = ["POST"])
def sudoku_api():
    if 'image' not in request.files:
        return jsonify({"error" : "Image not found"}) , 400
    
    file = request.files['image']
    image_bytes = file.read()

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    # build the grid
    grid = []
    numbers = ScanSudoku(img)
    count = 0
    for i in range(0,9):
        temp = []
        for j in range(0,9):
            temp.append(numbers[count])
            count +=1
        grid.append(temp)

    # Solve the grid
    is_solved = Solve(grid)
    mapper = lambda List: list(map(int , List))
    new_grid = [mapper(row) for row in grid]

    if is_solved:
        return jsonify({"solved" : "True" , "matrix" : new_grid})
    else:
        return jsonify({"solved" : "False"})

if __name__ == '__main__':
    app.run()


