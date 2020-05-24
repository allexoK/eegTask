import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy import signal

class featureExtractor:
    def __init__(self,hypnogramName,fileName):

        self.alphaTop = 10.5
        self.alphaBottom = 8
        self.thetaTop = 8
        self.thetaBottom = 4
        self.deltaTop = 4
        self.deltaBottom = 0.5

        self.data = pd.read_csv(fileName)
        self.data.columns = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T3","T4","T5","T6","FZ","CZ","PZ"]
        self.cols = self.data.columns
        self.hypns = pd.read_csv(hypnogramName).values
        self.alphaPowers = []
        self.alphaStds = []
        self.thetaPowers = []
        self.thetaStds = []
        self.deltaPowers = []
        self.deltaStds = []
        self.sleepPhases = []


    def getAlphaThetaDeltaPowers(self,p,fr):
        d = {}

        pNew = []
        fNew = []
        for j in range(len(p)):
            if(fr[j]>self.alphaBottom and fr[j]<self.alphaTop):
                pNew.append(p[j])
                fNew.append(fr[j])
        d["alphaMean"] = statistics.mean(pNew)
        d["alphaStd"] = statistics.stdev(pNew)

        pNew = []
        fNew = []
        for j in range(len(p)):
            if(fr[j]>self.thetaBottom and fr[j]<self.thetaTop):
                pNew.append(p[j])
                fNew.append(fr[j])
        d["thetaMean"] = statistics.mean(pNew)
        d["thetaStd"] = statistics.stdev(pNew)

        pNew = []
        fNew = []
        for j in range(len(p)):
            if(fr[j]>self.deltaBottom and fr[j]<self.deltaTop):
                pNew.append(p[j])
                fNew.append(fr[j])
        d["deltaMean"] = statistics.mean(pNew)
        d["deltaStd"] = statistics.stdev(pNew)

        return d


    def extractFeatures(self):
        # outputDataframe = pd.DataFrame()

        ctr = 0
        maxCtr = len(self.hypns)

        outputData = []
        for hypn in self.hypns:
            if(hypn[2] != 0): 
                oneSampleChar = []
                for name in range(1):
                    data = self.data[self.data.columns[name]][int(hypn[0]):int(hypn[1])]

                    ps = np.abs(np.fft.fft(data))**2
                    time_step = 1 / 250
                    freqs = np.fft.fftfreq(data.size, time_step)
                    idx = np.argsort(freqs)

                    fr = freqs[idx]
                    p = ps[idx] 
                    d = self.getAlphaThetaDeltaPowers(p,fr)
                    # oneSampleChar.append(d["alphaMean"])
                    oneSampleChar.append(d["alphaStd"])
                    # oneSampleChar.append(d["thetaMean"])
                    oneSampleChar.append(d["thetaStd"])
                    # oneSampleChar.append(d["deltaMean"])
                    oneSampleChar.append(d["deltaStd"])


                if(hypn[2] == 1):
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                if(hypn[2] == 2):
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                if(hypn[2] == 3):
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                if(hypn[2] == 4):
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                if(hypn[2] == 5):
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)
                outputData.append(oneSampleChar)
            print(ctr, " from ", maxCtr)
            ctr += 1

        outputDataframe = pd.DataFrame(data = outputData)
        outputDataframe.to_csv("analyzed.csv",index=False,header = False)


    def getSignalsToPredict(self):
        ctr = 0
        maxCtr = len(self.hypns)
        outputData = []
        outputNames = []
        for hypn in self.hypns:
            if(hypn[2] == 0): 
                data = self.data[self.data.columns[0]][int(hypn[0]):int(hypn[0])+7499]
                print(int(hypn[0]))

                fs = 250
                fc = 0.5  # Cut-off frequency of the filter
                w = fc / (fs / 2) # Normalize the frequency
                b, a = signal.butter(5, w, 'high')
                data = signal.filtfilt(b, a, data)

                fs = 250
                fc = 30  # Cut-off frequency of the filter
                w = fc / (fs / 2) # Normalize the frequency
                b, a = signal.butter(5, w, 'low')
                data = signal.filtfilt(b, a, data)

                # data1 = self.data[self.data.columns[0]][int(hypn[0]):int(hypn[0])+7499]

                # fs = 250
                # fc = 0.5  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(5, w, 'high')
                # data1 = signal.filtfilt(b, a, data1)

                # fs = 250
                # fc = 30  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(5, w, 'low')
                # data1 = signal.filtfilt(b, a, data1)

                # data2 = self.data[self.data.columns[1]][int(hypn[0]):int(hypn[0])+7499]

                # fs = 250
                # fc = 0.5  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(5, w, 'high')
                # data2 = signal.filtfilt(b, a, data2)

                # fs = 250
                # fc = 30  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(5, w, 'low')
                # data2 = signal.filtfilt(b, a, data2)

                # ps = np.abs(np.fft.fft(data1))**2
                # time_step = 1 / 250
                # freqs = np.fft.fftfreq(data1.size, time_step)
                # idx = np.argsort(freqs)
                # fr = freqs[idx]
                # p = ps[idx] 

                # data1 = p

                # ps = np.abs(np.fft.fft(data2))**2
                # time_step = 1 / 250
                # freqs = np.fft.fftfreq(data2.size, time_step)
                # idx = np.argsort(freqs)
                # fr = freqs[idx]
                # p = ps[idx] 

                # data2 = p

                # fs = 250
                # fc = 30  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(5, w, 'low')
                # data = signal.filtfilt(b, a, data)

                ps = np.abs(np.fft.fft(data))**2
                time_step = 1 / 250
                freqs = np.fft.fftfreq(data.size, time_step)
                idx = np.argsort(freqs)
                fr = freqs[idx]
                data = ps[idx] 

                # print("analyzed ",len(data))

                # data = list(map(lambda x: [x],data))
                # p = list(map(lambda x: [x],p))
                # out = []
                # for x in range(len(data)):
                #     out.append([data[x],p[x]])


                # d = []
                # for x in range(len(data)):
                #     d.append([data[x],p[x]])
                # outputData.append(d)


                # d = []
                # for x in range(len(data)):
                #     d.append([data1[x],data2[x]])
                # outputData.append(d)


                # newD = []
                # for j in range(len(data)):
                #     if(j%3 == 0):
                #         newD.append(data[j])
                # data = newD

                data = list(map(lambda x: [x],data))
                outputData.append(data)

                oneSampleChar = []
                if(hypn[2] == 1):
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                if(hypn[2] == 2):
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                if(hypn[2] == 3):
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                if(hypn[2] == 4):
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                if(hypn[2] == 5):
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)
                else:
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)

                outputNames.append(oneSampleChar)
            ctr += 1
        return [np.asarray(outputData),np.asarray(outputNames)]



    def getSignals(self):
        ctr = 0
        maxCtr = len(self.hypns)
        outputData = []
        outputNames = []
        
        for hypn in self.hypns:
            if(hypn[2] != 0): 
                data = self.data[self.data.columns[9]][int(hypn[0]):int(hypn[0])+7499]

                fs = 250
                fc = 0.5  # Cut-off frequency of the filter
                w = fc / (fs / 2) # Normalize the frequency
                b, a = signal.butter(5, w, 'high')
                data = signal.filtfilt(b, a, data)

                fs = 250
                fc = 30  # Cut-off frequency of the filter
                w = fc / (fs / 2) # Normalize the frequency
                b, a = signal.butter(5, w, 'low')
                data = signal.filtfilt(b, a, data)

                # data1 = self.data[self.data.columns[0]][int(hypn[0]):int(hypn[0])+7499]

                # fs = 250
                # fc = 0.5  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(5, w, 'high')
                # data1 = signal.filtfilt(b, a, data1)

                # fs = 250
                # fc = 30  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(5, w, 'low')
                # data1 = signal.filtfilt(b, a, data1)

                # data2 = self.data[self.data.columns[1]][int(hypn[0]):int(hypn[0])+7499]

                # fs = 250
                # fc = 0.5  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(5, w, 'high')
                # data2 = signal.filtfilt(b, a, data2)

                # fs = 250
                # fc = 30  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(5, w, 'low')
                # data2 = signal.filtfilt(b, a, data2)

                # ps = np.abs(np.fft.fft(data1))**2
                # time_step = 1 / 250
                # freqs = np.fft.fftfreq(data1.size, time_step)
                # idx = np.argsort(freqs)
                # fr = freqs[idx]
                # p = ps[idx] 

                # data1 = p

                # ps = np.abs(np.fft.fft(data2))**2
                # time_step = 1 / 250
                # freqs = np.fft.fftfreq(data2.size, time_step)
                # idx = np.argsort(freqs)
                # fr = freqs[idx]
                # p = ps[idx] 

                # data2 = p

                # fs = 250
                # fc = 30  # Cut-off frequency of the filter
                # w = fc / (fs / 2) # Normalize the frequency
                # b, a = signal.butter(5, w, 'low')
                # data = signal.filtfilt(b, a, data)

                ps = np.abs(np.fft.fft(data))**2
                time_step = 1 / 250
                freqs = np.fft.fftfreq(data.size, time_step)
                idx = np.argsort(freqs)
                fr = freqs[idx]
                data = ps[idx] 

                # print("analyzed ",len(data))

                # data = list(map(lambda x: [x],data))
                # p = list(map(lambda x: [x],p))
                # out = []
                # for x in range(len(data)):
                #     out.append([data[x],p[x]])


                # d = []
                # for x in range(len(data)):
                #     d.append([data[x],p[x]])
                # outputData.append(d)


                # d = []
                # for x in range(len(data)):
                #     d.append([data1[x],data2[x]])
                # outputData.append(d)


                # newD = []
                # for j in range(len(data)):
                #     if(j%3 == 0):
                #         newD.append(data[j])
                # data = newD

                data = list(map(lambda x: [x],data))
                outputData.append(data)

                oneSampleChar = []
                if(hypn[2] == 1):
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                if(hypn[2] == 2):
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                if(hypn[2] == 3):
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                if(hypn[2] == 4):
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)
                    oneSampleChar.append(0)
                if(hypn[2] == 5):
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(0)
                    oneSampleChar.append(1)

                outputNames.append(oneSampleChar)
            ctr += 1
        return [np.asarray(outputData),np.asarray(outputNames)]

    def updateFileWithNewValues(self,values,name):
        valCtr = 0
        for hypn in self.hypns:
            if(hypn[2] == 0): 
                if(values[valCtr][0] == 1):
                    hypn[2] = 1
                if(values[valCtr][1] == 1):
                    hypn[2] = 2
                if(values[valCtr][2] == 1):
                    hypn[2] = 3
                if(values[valCtr][3] == 1):
                    hypn[2] = 4
                if(values[valCtr][4] == 1):
                    hypn[2] = 5
                valCtr += 1
        df = pd.DataFrame(data = self.hypns)
        df.to_csv(name,index = False, header = False)