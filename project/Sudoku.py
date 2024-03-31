import pyautogui as pg
import time
import cv2 
import numpy as np
import os
from keras.models import load_model
################################# INPUT ########################################################################3
Imagepath = "/Users/Bingumalla Likith/Desktop/Projects/Sudoku_Solver/image.png"
height = 450
width = 450
threshold = 0.90
def preprocess(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converts the image to grey scale
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1) #add gaussian blurr
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,1,1,11,2) #Apply Adaptive threshold
    return imgThreshold

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)     #finding area of each contour
        if area > 50:
            peri = cv2.arcLength(i,True)   #storing dimensions
            approx =cv2.approxPolyDP(i,0.02*peri,True)      #counts the number of corners
            if area > max_area and len(approx) == 4:    #searches for all sqaures and rectangles
                biggest = approx    #contains all the corner points
                max_area = area     #contains the area of the region bounded by biggest
    return biggest,max_area

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2)) # 4*2 matrix formation
    myPointsNew = np.zeros((4,1,2), dtype = np.int32) #matrix with all elements as zeros
    
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def splitBoxes(img):
    row = np.vsplit(img,9)
    grid = []
    for r in row:
        col = np.hsplit(r,9)
        for box in col:
            grid.append(box)
    return grid

def imagescanning(boxes):
    numbers = []
    input = "/Users/Bingumalla Likith/Desktop/Projects/Sudoku_Solver/New_Model_trained.h5"
    model = load_model(input)

    print("Scanning Your Sudoku .....")
    for i in range(81):
        # Get the grid element
        grid_element = boxes[i]
    
        # Save the grid element as an image file
        filename = f"box_{i}.png"
        cv2.imwrite(filename, grid_element)
        # Read Image and Crop Borders
        img = cv2.imread(filename)
        os.remove(filename)
        img = np.asarray(img)
        img = cv2.resize(img,(32,32))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img / 255
        img = img.reshape(1,32,32,1)

        predictions = model.predict(img)
        prob = np.amax(predictions)
        num = np.argmax(predictions,axis=-1)[0]

        if(prob >= threshold):
            numbers.append(num)
        else:
            numbers.append(0)

    return numbers

#1.Prepare the image
img = cv2.imread(Imagepath)
img = cv2.resize(img,(width,height)) 
imgThreshold = preprocess(img)

#2.Find all contours
contours,hierarchy = cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#3.Find the biggest contour and use it as a sudoku
biggest,maxArea = biggestContour(contours)

if biggest.size != 0:
    biggest = reorder(biggest) #we need all the points to be in a proper order so we use this function
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0] , [width,0] , [0,height] , [width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarpColored = cv2.warpPerspective(img,matrix,(width,height)) #cropping of the required sudoku 
    imgWarpColored  = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) #convert it into grey scale

#4 Split each grid and find the number present in it
boxes = splitBoxes(imgWarpColored)
numbers = imagescanning(boxes)

###########################  SOLVING AND OUTPUT #################################################
grid = []

def output(grid):
    str_fin = []
    for i in grid:
        for j in i:
            str_fin.append(str(j))
    ready = input('\nAre you ready!!')
    if ready =='yes':
        time.sleep(10)
        for num in range(len(str_fin)):
            pg.press(str_fin[num])
            pg.hotkey('right')
            if (num+1)%9 == 0 :
                pg.hotkey('down')
                for i in range(9): pg.hotkey('left')

def present(c_row,c_col,x):
    #check if it is present in that column
    for i in range(0,9):
        if grid[i][c_col] == x  and i!=c_row:
            return False
    #check if it is present in that row
    for i in range(0,9):
        if grid[c_row][i] == x and i!=c_col:
            return False
    
    
    #check if it is present in the 3*3 grid
    grid_x = c_row//3 * 3
    grid_y = c_col//3 * 3

    for i in range(grid_x,grid_x + 3):
        for j in range(grid_y,grid_y + 3):
            if grid[i][j] == x:
                return False
    return True

#Applying backtracking algorithm
def solve(grid):
    for i in range(0,9):
        for j in range(0,9):
            if grid[i][j] == 0:
                for num in range(1,10):
                    if present(i,j,num):
                        grid[i][j] = num
                        solve(grid)
                        grid[i][j] = 0
                return 
    print('Your Ouput Sudoku ...')
    for i in grid:
        print(i)
    output(grid)

while(True):
    count = 0
    for i in range(0,9):
        temp = []
        for j in range(0,9):
           temp.append(numbers[count])
           count+=1
        grid.append(temp)
    break

print("")
print('Your input sudoku..')
for i in grid:
    print(i)

solve(grid)

                    
