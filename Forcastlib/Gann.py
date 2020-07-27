import pandas as pd
import numpy as np
import math
import time

class Boxes(object):#is the calls which allows for tuning params
    def __init__(self, name='test', ratio=0.5):
        self.name = name
        self.ratio = 0.5

    def Box(self, date, x, high, low, close, width):
        box = {
            'date' : date,
            'x1' : x,
            'x2' : x + width,
            'high' : high,
            'low' : low,
            'close' : close,
            'width' : width 
        }
        return box

    def make_boxes(self, data, inital = 0, inital_x = 0): #turns data into box format as a dict
        boxes = {}
        x = inital_x
        for i, val in enumerate(data['High']):
            count = inital + i
            width = (val - data['Low'][count]) * self.ratio
            boxes[i] = self.Box(data['Date'][count], x, val, data['Low'][count], data['Close'][count], width)
            x += width
        max_x = boxes[len(boxes)-1]['x2']
        return boxes, x        

    def check_vert(self, x1, x2, y1, y2, acc):#gets the graident of two points
        return round((y2-y1) / (x2-x1), acc)   

    def search(self, index, boxes, octaves, acc = 0):#]finds harmonics from one point
        box1 = boxes[index]#gets the box we check against
        ratios = pd.DataFrame(columns=['Box1 Date', 'Box2 Date', 'm'])#makes the result dataframe

        for i in range(0, index-1):
            box2 = boxes[i]
            m = self.Box_Grad(box1, box2, octaves, acc)
            row = [box1['Date'], box2['Date'], m]
            ratios.append(pd.Series(row, index=ratios.columns))
        return ratios

    def complete_search(self, boxes, octaves, acc=0):
        ratios = pd.DataFrame(columns=['Box1 Date', 'Box2 Date', 'm'])
        for i, val in enumerate(boxes):
            ratios.append(pd.Series(self.search(i, boxes, octaves, acc), ratios.columns))
        return ratios


    def HighLowSearch(self, boxes, count_pass, p=[0, "up"]):#this takes stock data not boxes
        count = 0
        last = {'High' : boxes[0]['high'], 'Low' : boxes[0]['low']}
        peaks = np.array([]).astype("int32")
        for i, x in enumerate(boxes):
            current = boxes[i]
            if (p[1] == "up" and last['High'] >= current['high']) or (p[1] == "down" and last['Low'] <= current['low']):
                count += 1
            else:
                count=0
            if count == count_pass:
                count = 0
                p[0] += 1
                if p[1] == "up":
                    p[1] = "down"
                    peaks = np.append(peaks, i-count_pass+1)
                else:
                    p[1] = "up"
                    peaks = np.append(peaks, i-count_pass+1)
            last = {'High' : current['high'], 'Low' : current['low']}
            
        return peaks

    def get_grades(self, peaks, boxes, octaves, acc):
        #grads = pd.DataFrame(columns=['Box1 Date', 'Box2 Date', 'm', 'B1 High', 'B2 High', 'B1 Low', 'B2 Low', 'B1 x1', 'B2 x1', 'B1 x2', 'B2 x2', 'B1 index', 'B2 index'])
        grads = []
        for i in range(0,len(peaks)-1):
            box1 = boxes[peaks[i]]
            box2 = boxes[peaks[i+1]]

            m = self.Box_Grad(box1, box2, octaves, acc)    
            if m != None:
                row = [box1['date'], box2['date'], m, box1['high'], box2['high'], box1['low'], box2['low'], box1['x1'], box2['x1'], box1['x2'], box2['x2'], int(peaks[i]), int(peaks[i+1])]
                grads.append(row)
        return grads

    def Box_Grad(self, box1, box2, octaves, acc=0):
        Box_arr = np.array([[box1['x1'], box1['high'], box2['x1'], box2['high']],[box1['x2'], box1['high'], box2['x1'], box2['high']], [box1['x2'], box1['low'], box2['x2'], box2['low']],[box1['x1'], box1['low'], box2['x1'], box2['low']], [box2['x1'], box2['low'], box1['x2'], box1['high']], [box2['x1'], box2['high'], box1['x2'], box2['low']]]).reshape((-1, 4))
        for i, x in enumerate(Box_arr):
            m = self.check_vert(x[0],x[2],x[1],x[3],acc)
            if m in octaves or m*-1 in octaves:
                return m
        return None

def peaks_to_values(peaks, boxes, start=1):
    temp = np.array([])
    for i, val in enumerate(peaks):
        if i % 2 == start:
            x = boxes[val]['high']

        else:
            x = boxes[val]['low']
        temp = np.append(temp, [x, boxes[val]['date']])

    return temp.reshape((-1,2))

def p_to_v(peaks, boxes, start=1):
    temp = np.array([])
    for i, val in enumerate(peaks):
        if i % 2 == start:
            x = boxes[val]['high']

        else:
            x = boxes[val]['low']
        temp = np.append(temp, [x])

    return temp


def peaks_to_dates(peaks, boxes):
    l = []
    for i in peaks:
        l.append([boxes[i]['date'], boxes[i]['high'], boxes[i]['low']])
    return l

def Grad_callback(data, params, initial):
    g = Boxes()
    try:
        boxes, x = g.make_boxes(data, initial['#start_index'], float(initial['#x']))
    except KeyError:
        boxes, x = g.make_boxes(data, initial['#start_index'])

    peaks = g.HighLowSearch(boxes, 3)
    grads = g.get_grades(peaks, boxes, params['octaves'], params['acc'])
    
    returns = {'!grads' : grads, 'peaks' : peaks, 'boxes' : boxes, '#x' : x, '#start_index' : data.index.max()}
    print(returns['#x'])
    return returns