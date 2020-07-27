import pandas_datareader as web
import numpy as np
import pandas as pd
import math
import json
import datetime

def Array_to_dict(data, labels):
    d = {}
    for i,row in enumerate(data):
        r = {}
        for j,x in enumerate(labels):
            r[x] = data[i][j]
        d[i] = r
    return d

def strDate_to_obj_datetime(date_array):#array of string dates
    temp = []
    for i, val in enumerate(date_array):
        temp.append(datetime.datetime.strptime(val, '%d/%m/%y %H:%M:%S' ))
    print(temp)
    return temp

def Date_Diff(date_array):#this need to be an array of datetime objects
    diff = np.array([])
    for i in range(0,len(date_array)-1):
        z = (date_array[i]-date_array[i+1]).days
        diff = np.append(diff, z)
    print(diff)
    return

def get_csv(filename, path='csv/'):
    return pd.read_csv(path+filename+'.csv')

def get_data(sym, data_source='yahoo'):#gets dataframe from datasource
    df = web.DataReader(sym, data_source=data_source)
    df['Date'] = df.index.astype(str)
    df = pd.DataFrame(df.values, columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'Date'])
    #df['index'] = list(range(0,df.shape[0]))
    #df.set_index('index')
    return df 

class Engine(object):
    def __init__(self, name='test'):
        """
        this is a data Engine which when given a dataset and a call back function,
         it will run a call back which is used to process data, the importance of this
        is that it allows my code to be run by steps and be ongoing instead of batch size

        the name is just for debugging an keeping track of instances
        """
        self.name = name
    
    def run(self, data, steps, callback, columns_names, return_labels, params=None):
        """
        this is where the data engine runs, 
        data is the dataset you want to run
        steps, how many times it saves the data while running.
        call back, the function which processes the data.
        the columns headings of the results table.
        params for the call back function for quick adjusting.
        """

        results = {'counters' : {}}#initalises the results object

        dataset_length = data.values.shape[0] 
        session_length = int(dataset_length / steps)
        print(dataset_length)
        print(session_length)
        print(steps)
        for step in range(0, steps):
            start_index = step * session_length
            end_index = (step+1) * session_length 
            session_data = data[start_index : end_index] 
            print(session_data)
            results['counters']['#start_index'] = session_data.index[0]
            returns = callback(session_data, params, results['counters'])
            results = self.return_handler(results, returns, return_labels, columns_names)
            self.store(results, self.name)
            print(f"{step+1} / {steps}")     

        return results

    def store(self, data, filename, path = "Data_pickle/"):#stores data into a pickle format ... might try json
        with open(path+filename+'.json', 'w+') as file:
            json.dump(data, file)

    def return_handler(self, results, return_dict, return_labels, columns_names):#turns the returns into the json format, if the key starts with a ! it turns the arr into a dict, if not it is just kept an array
        for i in return_labels:
            if i[0] == "!":
                d = Array_to_dict(return_dict[i], columns_names[i])
                try:
                    results[i].update(d)
                except KeyError:
                    results[i] = d
            elif i[0] == "#": #checks for a counter
                results['counters'][i] = return_dict[i]
            else:
                if isinstance(return_dict[i], np.ndarray):
                    return_dict[i] = return_dict[i].tolist()
                try:
                    results[i].extend(return_dict[i])
                except AttributeError:
                    results[i] = return_dict[i]
                except KeyError:
                    results[i] = return_dict[i]

        return results



def csv_destro(csv_name):
    df = pd.read_csv(csv_name)
    last_row = []
    Data = df.values
    #print(Data)

    temp = []
    for row in Data.tolist():
        #print(row[1])
        if math.isnan(row[1]):
            #print(row[1])
            #row = [row[0], last_row[1], last_row[2], last_row[3]]
            pass
        else:
            last_row = row
            #print(row)
            temp.append(row)

    #print(temp)

    df = pd.DataFrame(temp, columns=['Date', 'High', 'Low', 'Close'])
    return df

def csv_clean(csv_name):
    df = pd.read_csv(csv_name)
    last_row = []
    Data = df.values
    #print(Data)

    temp = []
    for row in Data.tolist():
        #print(row[1])
        if math.isnan(row[1]):
            #print(row[1])
            row = [row[0], last_row[1], last_row[2], last_row[3]]
        last_row = row
        #print(row)
        temp.append(row)

    #print(temp)

    df = pd.DataFrame(temp, columns=['Date', 'High', 'Low', 'Close'])
    return df
