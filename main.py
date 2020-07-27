import Datalib.DataManage as DM
import Forcastlib.Gann as Gann
import Forcastlib.seqeunce as Seq
import Forcastlib.CNN as CNN
import pandas as pd
import matplotlib.pyplot as plt



def main():
    columns = {'!grads' : ['Box1 Date', 'Box2 Date', 'm', 'B1 High', 'B2 High', 'B1 Low', 'B2 Low', 'B1 x1', 'B2 x1', 'B1 x2', 'B2 x2', 'B1 index', 'B2 index']}#each columns_label set has a key for multiple returns
    octaves = pd.read_csv('csv/Seq.csv')['z(ratio)']
    params = {'acc' : 4, 'octaves' : octaves}
    #df = DM.get_data('^DJI')
    #df = DM.get_csv('mData')
    #df = DM.get_data('^DJI')
    df = DM.get_csv('mData')
    eng = DM.Engine('test')
    r = eng.run(df, 1, Gann.Grad_callback, columns, ['!grads', 'peaks', 'boxes', '#start_index', '#x'], params)
    gradsdf = pd.DataFrame.from_dict(r['!grads'], orient='index')
    print(gradsdf)

    peaks = Gann.p_to_v(r['peaks'], r['boxes'])
    #peaksdf = pd.DataFrame(peaks.reshape((-1,2)), columns=['Value', 'Date'])
    #print(peaksdf)

    c = CNN.HighLowAI(5,255, 2)
    c.run(peaks, 15)
    maxi = len(peaks) - 1
    arr = [peaks[maxi-2], peaks[maxi-1]]
    p = c.predict(arr)
    print("prediciton : ")
    print(p)
    print(peaks[maxi])

    
if __name__ == "__main__":
    main()