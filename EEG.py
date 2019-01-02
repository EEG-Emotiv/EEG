import sys
sys.path.insert(0, "./src")
from emokit.emotiv import Emotiv
from emokit.packet import EmotivExtraPacket
from emokit.util import get_quality_scale_level_color
import datetime
from Coefi_Data import *
import time
from sklearn.preprocessing import minmax_scale
import pywt
from hpelm import ELM
from pywt import wavedec
from pywt import upcoef
import numpy as np
import threading
from scipy.signal import butter, lfilter
import gevent
import os
import GUI_EEG
from PyQt4 import QtGui,QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import serial



sampling_rate = 128  #In hertz
number_of_channel = 13

channel_names =[
	"F3" ,
    "FC5",
    "F7",
	"T7",
	"P7",
	"O1",
	"O2",
	"P8",
	"T8",
	"FC6",
	"F4",
	"F8",
	"AF4"
]
def D_bandpass(lowcut,hightcut,fs,order):
    # Membuat Desain Filter
    nyq = 0.5*fs
    low = lowcut/nyq
    high = hightcut/nyq
    b,a = butter(order,[low, high],btype='band')
    return b,a
def F_bandpass(data, lowcut,hightcut,fs,order):
    b,a = D_bandpass(lowcut,hightcut,fs,order=order)
    y = lfilter(b,a,data)
    return y
def wrcoef(X, coef_type, coeffs,wavename,level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return upcoef('a',a,wavename,level=level)[:N]
    elif coef_type =='d':
        return upcoef('d',ds[level-1], wavename,level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))
class Realtime(object) :
    def __init__(self, port,dir,baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.direct = dir
    def Wavelet(self,data):
        coeffs1 = wavedec(data[0], 'db4', level=4)
        coeffs2 = wavedec(data[1], 'db4', level=4)
        coeffs3 = wavedec(data[2], 'db4', level=4)
        coeffs4 = wavedec(data[3], 'db4', level=4)
        coeffs5 = wavedec(data[4], 'db4', level=4)
        coeffs6 = wavedec(data[5], 'db4', level=4)
        coeffs7 = wavedec(data[6], 'db4', level=4)
        coeffs8 = wavedec(data[7], 'db4', level=4)
        coeffs9 = wavedec(data[8], 'db4', level=4)
        coeffs10 = wavedec(data[9], 'db4', level=4)
        coeffs11 = wavedec(data[10], 'db4', level=4)
        coeffs12 = wavedec(data[11], 'db4', level=4)
        coeffs13 = wavedec(data[12], 'db4', level=4)
        # Koefisien Filter BP
        lowcut1 = 8.0
        highcut1 = 16.0
        lowcut2 = 16.0
        highcut2 = 24.0
        fs = 128
        # # # # # # # # # # #
        # Wr Coeficient dan Hasil Filter dan Average
        Alpha1 = F_bandpass((wrcoef(data[0], 'd', coeffs1, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha2 = F_bandpass((wrcoef(data[1], 'd', coeffs2, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha3 = F_bandpass((wrcoef(data[2], 'd', coeffs3, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha4 = F_bandpass((wrcoef(data[3], 'd', coeffs4, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha5 = F_bandpass((wrcoef(data[4], 'd', coeffs5, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha6 = F_bandpass((wrcoef(data[5], 'd', coeffs6, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha7 = F_bandpass((wrcoef(data[6], 'd', coeffs7, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha8 = F_bandpass((wrcoef(data[7], 'd', coeffs8, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha9 = F_bandpass((wrcoef(data[8], 'd', coeffs9, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha10 = F_bandpass((wrcoef(data[9], 'd', coeffs10, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha11 = F_bandpass((wrcoef(data[10], 'd', coeffs11, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha12 = F_bandpass((wrcoef(data[11], 'd', coeffs12, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha13 = F_bandpass((wrcoef(data[12], 'd', coeffs13, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)

        Betha1 = F_bandpass((wrcoef(data[0], 'd', coeffs1, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha2 = F_bandpass((wrcoef(data[1], 'd', coeffs2, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha3 = F_bandpass((wrcoef(data[2], 'd', coeffs3, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha4 = F_bandpass((wrcoef(data[3], 'd', coeffs4, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha5 = F_bandpass((wrcoef(data[4], 'd', coeffs5, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha6 = F_bandpass((wrcoef(data[5], 'd', coeffs6, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha7 = F_bandpass((wrcoef(data[6], 'd', coeffs7, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha8 = F_bandpass((wrcoef(data[7], 'd', coeffs8, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha9 = F_bandpass((wrcoef(data[8], 'd', coeffs9, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha10 = F_bandpass((wrcoef(data[9], 'd', coeffs10, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha11 = F_bandpass((wrcoef(data[10], 'd', coeffs11, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha12 = F_bandpass((wrcoef(data[11], 'd', coeffs12, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha13 = F_bandpass((wrcoef(data[12], 'd', coeffs13, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        PS_Alpha[0] = np.average((np.absolute((np.fft.fft(Alpha1))) ** 2))
        PS_Alpha[1] = np.average((np.absolute((np.fft.fft(Alpha2))) ** 2))
        PS_Alpha[2] = np.average((np.absolute((np.fft.fft(Alpha3))) ** 2))
        PS_Alpha[3] = np.average((np.absolute((np.fft.fft(Alpha4))) ** 2))
        PS_Alpha[4] = np.average((np.absolute((np.fft.fft(Alpha5))) ** 2))
        PS_Alpha[5] = np.average((np.absolute((np.fft.fft(Alpha6))) ** 2))
        PS_Alpha[6] = np.average((np.absolute((np.fft.fft(Alpha7))) ** 2))
        PS_Alpha[7] = np.average((np.absolute((np.fft.fft(Alpha8))) ** 2))
        PS_Alpha[8] = np.average((np.absolute((np.fft.fft(Alpha9))) ** 2))
        PS_Alpha[9] = np.average((np.absolute((np.fft.fft(Alpha10))) ** 2))
        PS_Alpha[10] = np.average((np.absolute((np.fft.fft(Alpha11))) ** 2))
        PS_Alpha[11] = np.average((np.absolute((np.fft.fft(Alpha12))) ** 2))
        PS_Alpha[12] = np.average((np.absolute((np.fft.fft(Alpha13))) ** 2))

        PS_Betha[0] = np.average((np.absolute((np.fft.fft(Betha1))) ** 2))
        PS_Betha[1] = np.average((np.absolute((np.fft.fft(Betha2))) ** 2))
        PS_Betha[2] = np.average((np.absolute((np.fft.fft(Betha3))) ** 2))
        PS_Betha[3] = np.average((np.absolute((np.fft.fft(Betha4))) ** 2))
        PS_Betha[4] = np.average((np.absolute((np.fft.fft(Betha5))) ** 2))
        PS_Betha[5] = np.average((np.absolute((np.fft.fft(Betha6))) ** 2))
        PS_Betha[6] = np.average((np.absolute((np.fft.fft(Betha7))) ** 2))
        PS_Betha[7] = np.average((np.absolute((np.fft.fft(Betha8))) ** 2))
        PS_Betha[8] = np.average((np.absolute((np.fft.fft(Betha9))) ** 2))
        PS_Betha[9] = np.average((np.absolute((np.fft.fft(Betha10))) ** 2))
        PS_Betha[10] = np.average((np.absolute((np.fft.fft(Betha11))) ** 2))
        PS_Betha[11] = np.average((np.absolute((np.fft.fft(Betha12))) ** 2))
        PS_Betha[12] = np.average((np.absolute((np.fft.fft(Betha13))) ** 2))

        PS1 = np.concatenate(([PS_Alpha[0]], [PS_Betha[0]]), axis=0)
        PS2 = np.concatenate(([PS_Alpha[1]], [PS_Betha[1]]), axis=0)
        PS3 = np.concatenate(([PS_Alpha[2]], [PS_Betha[2]]), axis=0)
        PS4 = np.concatenate(([PS_Alpha[3]], [PS_Betha[3]]), axis=0)
        PS5 = np.concatenate(([PS_Alpha[4]], [PS_Betha[4]]), axis=0)
        PS6 = np.concatenate(([PS_Alpha[5]], [PS_Betha[5]]), axis=0)
        PS7 = np.concatenate(([PS_Alpha[6]], [PS_Betha[6]]), axis=0)
        PS8 = np.concatenate(([PS_Alpha[7]], [PS_Betha[7]]), axis=0)
        PS9 = np.concatenate(([PS_Alpha[8]], [PS_Betha[8]]), axis=0)
        PS10 = np.concatenate(([PS_Alpha[9]], [PS_Betha[9]]), axis=0)
        PS11 = np.concatenate(([PS_Alpha[10]], [PS_Betha[10]]), axis=0)
        PS12 = np.concatenate(([PS_Alpha[11]], [PS_Betha[11]]), axis=0)
        PS13 = np.concatenate(([PS_Alpha[12]], [PS_Betha[12]]), axis=0)

        #############################################################################################

        y1 = np.concatenate((PS1, PS2, PS3, PS4, PS5, PS6, PS7, PS8, PS9, PS10, PS11, PS12, PS13), axis=0)

        return y1
    def Normalisasi (self,N):
        return minmax_scale(N, feature_range=(0, 1))
    def ELM(self,Z):
        Z1 = Z.reshape(1, 26)
        elm = ELM(Z1.shape[1], 2)
        direct_open = self.direct
        elm.load(direct_open)
        Y1 = elm.predict(Z1)

        n = len(Y1)
        m1 = []
        for i in range(0, n):
            maximal1 = np.max(Y1[i, :])
            m1.append(maximal1)

        Y21 = []
        for i in range(0, n):
            if (Y1[i, 0] == m1[i]):
                Y21.append([1, 0, 0])
                a = 1

            elif (Y1[i, 1] == m1[i]):
                Y21.append([0, 1, 0])
                a = 2

            elif (Y1[i, 2] == m1[i]):
                Y21.append([0, 0, 1])
                a = 3

        return a
    def Kirim_Arduino(self,Y):
        from time import sleep
        import serial

        Ready_flag = 0
        portname = self.port
        baud = self.baudrate
        ser = serial.Serial(portname, baud, timeout=2)
        def SerialWrite(command):
            ser.write(command)
            rv = ser.readline()
            sleep(1)
            ser.flushInput()
        if (Y == 1):  # if the value is 1
            Arduino_cmd = '1'
            print("maju")
        elif (Y == 2):  # if the value is 0
            Arduino_cmd = '2'
            print("mundur")
        elif (Y == 3):
            Arduino_cmd = '3'
            print("Berhenti")
        cmd = Arduino_cmd.encode("utf-8")
        SerialWrite(Arduino_cmd)
    def process_all_data(self,all_channel_data):
        # Get Normalisasi
        feature = self.Wavelet(all_channel_data)
        # Get Normalisasi
        Norm = self.Normalisasi(feature)
        # Predict movement
        a = self.ELM(Norm)
        # control Arduino
        self.Kirim_Arduino(a)
    def main_process(self):
         threads = []
         eeg_realtime = np.zeros((number_of_channel, number_of_realtime_eeg), dtype=np.double)
         counter = 0
         # endTime = datetime.datetime.now() + datetime.timedelta(seconds=realtime_eeg_in_second)
         with Emotiv(display_output=False) as headset:
             while headset.running:
                 # Looping to get realtime EEG data from Emotiv EPOC
                 try:
                     packet = headset.dequeue()
                     if packet is not None:
                         if type(packet) != EmotivExtraPacket:
                             if init:
                                 for i in range(number_of_channel):
                                     eeg_realtime[i, counter] = packet.sensors[channel_names[i]]['value']
                             else:
                                 for i in range(number_of_channel):
                                     new_data = [packet.sensors[channel_names[i]]['value']]
                                     eeg_realtime = np.insert(eeg_realtime, number_of_realtime_eeg, new_data, axis=1)
                                     eeg_realtime = np.delete(eeg_realtime, 0, axis=1)

                             # If EEG data have been recorded in ... seconds, then process data to predict movement
                             # if counter == (sampling_rate - 1) or counter == (number_of_realtime_eeg - 1):
                             if counter == (number_of_realtime_eeg - 1):
                             # if datetime.datetime.now() == endTime:
                                 t = threading.Thread(target=rte.process_all_data, args=(eeg_realtime,))
                                 threads.append(t)
                                 t.start()
                                 # endTime = datetime.datetime.now() + datetime.timedelta(seconds=realtime_eeg_in_second)
                                 counter = 0


                             counter += 1

                 except KeyboardInterrupt:
                     import os
                 finally:
                     import os
class Training_Data(object):
    global P,R
    def __init__(self, data_Train,data_Train_target,data_Test,data_Test_target,direct,e):
        self.data_Train = data_Train
        self.data_Train_target = data_Train_target
        self.data_Test = data_Test
        self.data_Test_target = data_Test_target
        self.direct = direct
        self.e = e
    def Wavelet_TD(self,data):
        coeffs1 = wavedec(data[0], 'db4', level=4)
        coeffs2 = wavedec(data[1], 'db4', level=4)
        coeffs3 = wavedec(data[2], 'db4', level=4)
        coeffs4 = wavedec(data[3], 'db4', level=4)
        coeffs5 = wavedec(data[4], 'db4', level=4)
        coeffs6 = wavedec(data[5], 'db4', level=4)
        coeffs7 = wavedec(data[6], 'db4', level=4)
        coeffs8 = wavedec(data[7], 'db4', level=4)
        coeffs9 = wavedec(data[8], 'db4', level=4)
        coeffs10 = wavedec(data[9], 'db4', level=4)
        coeffs11 = wavedec(data[10], 'db4', level=4)
        coeffs12 = wavedec(data[11], 'db4', level=4)
        coeffs13 = wavedec(data[12], 'db4', level=4)
        # Koefisien Filter BP
        lowcut1 = 8.0
        highcut1 = 16.0
        lowcut2 = 16.0
        highcut2 = 24.0
        fs = 128
        # # # # # # # # # # #
        # Wr Coeficient dan Hasil Filter dan Average
        Alpha1 = F_bandpass((wrcoef(data[0], 'd', coeffs1, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha2 = F_bandpass((wrcoef(data[1], 'd', coeffs2, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha3 = F_bandpass((wrcoef(data[2], 'd', coeffs3, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha4 = F_bandpass((wrcoef(data[3], 'd', coeffs4, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha5 = F_bandpass((wrcoef(data[4], 'd', coeffs5, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha6 = F_bandpass((wrcoef(data[5], 'd', coeffs6, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha7 = F_bandpass((wrcoef(data[6], 'd', coeffs7, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha8 = F_bandpass((wrcoef(data[7], 'd', coeffs8, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha9 = F_bandpass((wrcoef(data[8], 'd', coeffs9, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha10 = F_bandpass((wrcoef(data[9], 'd', coeffs10, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha11 = F_bandpass((wrcoef(data[10], 'd', coeffs11, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha12 = F_bandpass((wrcoef(data[11], 'd', coeffs12, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)
        Alpha13 = F_bandpass((wrcoef(data[12], 'd', coeffs13, 'db4', 3))
                              , lowcut1, highcut1, fs, order=6)

        Betha1 = F_bandpass((wrcoef(data[0], 'd', coeffs1, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha2 = F_bandpass((wrcoef(data[1], 'd', coeffs2, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha3 = F_bandpass((wrcoef(data[2], 'd', coeffs3, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha4 = F_bandpass((wrcoef(data[3], 'd', coeffs4, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha5 = F_bandpass((wrcoef(data[4], 'd', coeffs5, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha6 = F_bandpass((wrcoef(data[5], 'd', coeffs6, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha7 = F_bandpass((wrcoef(data[6], 'd', coeffs7, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha8 = F_bandpass((wrcoef(data[7], 'd', coeffs8, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha9 = F_bandpass((wrcoef(data[8], 'd', coeffs9, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha10 = F_bandpass((wrcoef(data[9], 'd', coeffs10, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha11 = F_bandpass((wrcoef(data[10], 'd', coeffs11, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha12 = F_bandpass((wrcoef(data[11], 'd', coeffs12, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        Betha13 = F_bandpass((wrcoef(data[12], 'd', coeffs13, 'db4', 2))
                              , lowcut2, highcut2, fs, order=6)
        PS_Alpha[0] = np.average((np.absolute((np.fft.fft(Alpha1))) ** 2))
        PS_Alpha[1] = np.average((np.absolute((np.fft.fft(Alpha2))) ** 2))
        PS_Alpha[2] = np.average((np.absolute((np.fft.fft(Alpha3))) ** 2))
        PS_Alpha[3] = np.average((np.absolute((np.fft.fft(Alpha4))) ** 2))
        PS_Alpha[4] = np.average((np.absolute((np.fft.fft(Alpha5))) ** 2))
        PS_Alpha[5] = np.average((np.absolute((np.fft.fft(Alpha6))) ** 2))
        PS_Alpha[6] = np.average((np.absolute((np.fft.fft(Alpha7))) ** 2))
        PS_Alpha[7] = np.average((np.absolute((np.fft.fft(Alpha8))) ** 2))
        PS_Alpha[8] = np.average((np.absolute((np.fft.fft(Alpha9))) ** 2))
        PS_Alpha[9] = np.average((np.absolute((np.fft.fft(Alpha10))) ** 2))
        PS_Alpha[10] = np.average((np.absolute((np.fft.fft(Alpha11))) ** 2))
        PS_Alpha[11] = np.average((np.absolute((np.fft.fft(Alpha12))) ** 2))
        PS_Alpha[12] = np.average((np.absolute((np.fft.fft(Alpha13))) ** 2))

        PS_Betha[0] = np.average((np.absolute((np.fft.fft(Betha1))) ** 2))
        PS_Betha[1] = np.average((np.absolute((np.fft.fft(Betha2))) ** 2))
        PS_Betha[2] = np.average((np.absolute((np.fft.fft(Betha3))) ** 2))
        PS_Betha[3] = np.average((np.absolute((np.fft.fft(Betha4))) ** 2))
        PS_Betha[4] = np.average((np.absolute((np.fft.fft(Betha5))) ** 2))
        PS_Betha[5] = np.average((np.absolute((np.fft.fft(Betha6))) ** 2))
        PS_Betha[6] = np.average((np.absolute((np.fft.fft(Betha7))) ** 2))
        PS_Betha[7] = np.average((np.absolute((np.fft.fft(Betha8))) ** 2))
        PS_Betha[8] = np.average((np.absolute((np.fft.fft(Betha9))) ** 2))
        PS_Betha[9] = np.average((np.absolute((np.fft.fft(Betha10))) ** 2))
        PS_Betha[10] = np.average((np.absolute((np.fft.fft(Betha11))) ** 2))
        PS_Betha[11] = np.average((np.absolute((np.fft.fft(Betha12))) ** 2))
        PS_Betha[12] = np.average((np.absolute((np.fft.fft(Betha13))) ** 2))

        PS1 = np.concatenate(([PS_Alpha[0]], [PS_Betha[0]]), axis=0)
        PS2 = np.concatenate(([PS_Alpha[1]], [PS_Betha[1]]), axis=0)
        PS3 = np.concatenate(([PS_Alpha[2]], [PS_Betha[2]]), axis=0)
        PS4 = np.concatenate(([PS_Alpha[3]], [PS_Betha[3]]), axis=0)
        PS5 = np.concatenate(([PS_Alpha[4]], [PS_Betha[4]]), axis=0)
        PS6 = np.concatenate(([PS_Alpha[5]], [PS_Betha[5]]), axis=0)
        PS7 = np.concatenate(([PS_Alpha[6]], [PS_Betha[6]]), axis=0)
        PS8 = np.concatenate(([PS_Alpha[7]], [PS_Betha[7]]), axis=0)
        PS9 = np.concatenate(([PS_Alpha[8]], [PS_Betha[8]]), axis=0)
        PS10 = np.concatenate(([PS_Alpha[9]], [PS_Betha[9]]), axis=0)
        PS11 = np.concatenate(([PS_Alpha[10]], [PS_Betha[10]]), axis=0)
        PS12 = np.concatenate(([PS_Alpha[11]], [PS_Betha[11]]), axis=0)
        PS13 = np.concatenate(([PS_Alpha[12]], [PS_Betha[12]]), axis=0)

        #############################################################################################

        y1 = np.concatenate((PS1, PS2, PS3, PS4, PS5, PS6, PS7, PS8, PS9, PS10, PS11, PS12, PS13), axis=0)

        return y1
    def Normalisasi_TD (self,N):
        return minmax_scale(N, feature_range=(0, 1))
    def ELM_TD(self):
        global P,R
        X = self.data_Train
        T = self.data_Train_target
        Z = self.data_Test
        T1 = self.data_Test_target
        direct_save = str(self.direct)
        error = self.e
        R = 1
        while (R >= error):
            elm = ELM(X.shape[1], T.shape[1])
            elm.add_neurons(20, "sigm")  # variasi hidden neuron
            elm.train(X, T, "c")
            Y = elm.predict(X)  # Untuk Prediksi
            P = elm.error(T, Y)  # Latih dan Uji

            n = len(Y)
            m = []
            for i in range(0, n):
                maximal = np.max(Y[i, :])
                m.append(maximal)

            Y2 = []
            for i in range(0, n):
                if (Y[i, 0] == m[i]):
                    Y2.append([1, 0, 0])

                elif (Y[i, 1] == m[i]):

                    Y2.append([0, 1, 0])
                elif (Y[i, 2] == m[i]):
                    Y2.append([0, 0, 1])

            elm.save(direct_save)

            # Testing

            elm = ELM(Z.shape[1], T1.shape[1])
            elm.load(direct_save)
            Y1 = elm.predict(Z)
            R = elm.error(T1, Y1)  # Prediksi

            n = len(Y1)
            m1 = []
            for i in range(0, n):
                maximal1 = np.max(Y1[i, :])
                m1.append(maximal1)

            Y21 = []
            for i in range(0, n):
                if (Y1[i, 0] == m1[i]):
                    Y21.append([1, 0, 0])
                elif (Y1[i, 1] == m1[i]):
                    Y21.append([0, 1, 0])
                elif (Y1[i, 2] == m1[i]):
                    Y21.append([0, 0, 1])
        return R
    def process_data(self,hasil):
        # Predict movement
        P,R = self.ELM_TD(hasil)
        return P,R
    def Kumpul_Data(self):
         threads = []
         eeg_realtime = np.zeros((number_of_channel, number_of_realtime_eeg), dtype=np.double)
         counter = 0
         # endTime = datetime.datetime.now() + datetime.timedelta(seconds=realtime_eeg_in_second)
         with Emotiv(display_output=False) as headset:
             while headset.running:
                 # Looping to get realtime EEG data from Emotiv EPOC
                 try:
                     packet = headset.dequeue()
                     if packet is not None:
                         if type(packet) != EmotivExtraPacket:
                             if init:
                                 for i in range(number_of_channel):
                                     eeg_realtime[i, counter] = packet.sensors[channel_names[i]]['value']
                             else:
                                 for i in range(number_of_channel):
                                     new_data = [packet.sensors[channel_names[i]]['value']]
                                     eeg_realtime = np.insert(eeg_realtime, number_of_realtime_eeg, new_data, axis=1)
                                     eeg_realtime = np.delete(eeg_realtime, 0, axis=1)

                             # If EEG data have been recorded in ... seconds, then process data to predict movement
                             # if counter == (sampling_rate - 1) or counter == (number_of_realtime_eeg - 1):
                             if counter == (number_of_realtime_eeg - 1):
                             # if datetime.datetime.now() == endTime:
                             # Get Normalisasi
                                feature = self.Wavelet_TD(all_channel_data)
                             # Get Normalisasi
                                Norm = self.Normalisasi_TD(feature)
                                headset.quit()
                             counter += 1

                 except KeyboardInterrupt:
                     import os
                 finally:
                     import os
         return Norm
class Cek_Robot(object):
    def Kirim_Arduino(self,Y,port,baudrate):
        from time import sleep
        import serial

        Ready_flag = 0
        ser = serial.Serial(port, baudrate, timeout=2)
        def SerialWrite(command):
            ser.write(command)
            sleep(1)
        if (Y == 1):  # if the value is 1
            Arduino_cmd = '1'
            print("maju")
        elif (Y == 2):  # if the value is 0
            Arduino_cmd = '2'
            print("mundur")
        elif (Y == 3):
            Arduino_cmd = '3'
            print("Berhenti")
        # cmd = Arduino_cmd.encode("utf-8")
        SerialWrite(Arduino_cmd)
class Cek_Emotiv(object):
    def Cek(self):
        # -*- coding: utf-8 -*-
        # This is an example of popping a packet from the Emotiv class's packet queue
        from emokit.emotiv import Emotiv
        if __name__ == "__main__":
            with Emotiv(display_output=True, verbose=True) as headset:
                packet = headset.dequeue()
                if packet is not None:
                    pass

Robot = False
Training = False
Integrasi = False
count = 0
a= 0
b=0
msg_Tar_Tra = ""
msg_Tar_Tes = ""




if __name__ == "__main__":
    global P,R
    app = QtGui.QApplication(sys.argv)
    form = GUI_EEG.QtGui.QMainWindow()
    gui = GUI_EEG.Ui_MainWindow()
    gui.setupUi(form)
    gui.Total_Training.setMinimum(1)
    gui.Total_Testing.setMinimum(1)
    D_Training = gui.Total_Training.text()
    D_Testing = gui.Total_Testing.text()
    # GUI Code #
    #####################################################################
    # Open Dialog
    def Direct():
        path = str(QtGui.QFileDialog.getSaveFileName(None,'Select a folder to Save File'))
        gui.Directory_Line.setText(path)
    gui.OpenDialog.clicked.connect(Direct)
    # Button Cek
    def Cek():
        if __name__ == "__main__":
            Test = Cek_Emotiv()
            Test.Cek()
    gui.CEK.clicked.connect(Cek)
    # Button Exit
    def Close():
        import os
        os._exit(0)
    gui.EXIT.clicked.connect(Close)
    # Button Eksekusi
    def serial_ports():
        """ Lists serial port names

            :raises EnvironmentError:
                On unsupported or unknown platforms
            :returns:
                A list of the serial ports available on the system
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        else:
            raise EnvironmentError('Unsupported platform')
        return ports
    def Eksekusi():
        global Robot,Integrasi,Training,D_Training,D_Testing,Kumpul_D_Testing,Kumpul_D_Training\
            ,Kumpul_D_Target_Testing,Kumpul_D_Target_Training,realtime_eeg_in_second,number_of_realtime_eeg
        if int(int(gui.time.text()))>0:
            # Koefisien Waktu
            realtime_eeg_in_second = float(gui.time.text())
            number_of_realtime_eeg = sampling_rate * realtime_eeg_in_second
            if gui.radio_CekRobot.isChecked():
                Robot = True
                Integrasi = False
                Training = False
                ports = serial_ports()
                for port in ports:
                    try:
                        s = serial.Serial(port)
                        s.close()
                        gui.COM.addItem(port)
                    except (OSError, serial.SerialException):
                        pass
            elif gui.radio_Training.isChecked():
                Training = True
                Integrasi = False
                Robot = False
                gui.Table_Training.setRowCount(int(D_Training))
                gui.Table_Testing.setRowCount(int(D_Testing))
                gui.Table_Target_Testing.setRowCount(int(D_Testing))
                gui.Table_Target_Training.setRowCount(int(D_Training))
                Kumpul_D_Training = np.zeros(((number_of_channel * 2), int(D_Training)), dtype=np.double)
                Kumpul_D_Target_Training = np.zeros(((number_of_channel * 2), int(D_Training)), dtype=np.double)
                Kumpul_D_Target_Testing = np.zeros(((number_of_channel * 2), int(D_Testing)), dtype=np.double)
            elif gui.radio_Integrasi.isChecked():
                Integrasi = True
                Training = False
                Robot = False
                ports = serial_ports()
                for port in ports:
                    try:
                        s = serial.Serial(port)
                        s.close()
                        gui.COM.addItem(port)
                    except (OSError, serial.SerialException):
                        pass
            else:

                msg = QMessageBox()
                msg.setText("Pilih yang ingin dipilih : Training, Cek Robot, Integrasi")
                msg.exec_()
        else:

            msg = QMessageBox()
            msg.setText("Silakan Isi Waktu Pengambilan Data")
            msg.exec_()

    gui.EKSEKUSI.clicked.connect(Eksekusi)
    # Button Reset
    def restart_program():
        """Restarts the current program.
        Note: this function does not return. Any cleanup action (like
        saving data) must be done before calling this function."""
        python = sys.executable
        os.execl(python, python, *sys.argv)
    gui.RESET.clicked.connect(restart_program)
    # CEK ROBOT
    def Maju():
        global Robot
        if __name__ == "__main__":
            if Robot:
                port = gui.COM.currentText()
                M_aju = Cek_Robot()
                M_aju.Kirim_Arduino(1,port=str(port),baudrate=9600)
    gui.Maju_Cek.clicked.connect(Maju)
    def Mundur():
        global Robot
        if __name__ == "__main__":
            if Robot:
                port = gui.COM.currentText()
                M_undur = Cek_Robot()
                M_undur.Kirim_Arduino(2,port=str(port),baudrate=9600)
    gui.Mundur_Cek.clicked.connect(Mundur)
    def Berhenti():
        global Robot
        if __name__ == "__main__":
            if Robot:
                port = gui.COM.currentText()
                B_erhenti = Cek_Robot()
                B_erhenti.Kirim_Arduino(3,port=str(port),baudrate=9600)
    gui.Berhenti_Cek.clicked.connect(Berhenti)
    # Training Data
    # Ambil Data
    def Collect_Data():
        global Hasil,count,realtime_eeg_in_second,number_of_realtime_eeg
        if __name__ == "__main__":
            Coll = Training_Data()
            Hasil = Coll.Kumpul_Data()
            if (count<D_Training):
                for i in range(0,(number_of_channel*2)-1):
                    gui.Table_Training.setItem(count,i,QtGui.QTableWidgetItem(str(Hasil[i])))
            elif ((count>=D_Training) and (count<D_Testing)):
                for i in range(0,(number_of_channel*2)-1):
                    gui.Table_Testing.setItem(count,i,QtGui.QTableWidgetItem(str(Hasil[i])))
            count +=1
    gui.AmbilData.clicked.connect(Collect_Data)
    # Add Data To Array
    def Add():
        global Kumpul_D_Testing,Kumpul_D_Training,Hasil,Kumpul_D_Target_Testing,Kumpul_D_Target_Training
        count_add = count - 1
        for j in range(3):
            if gui.Table_Target_Training.item(count_add,j) != "":
                a +=1
            else:
                a +=0
                msg_Tar_Tra = msg_Tar_Tra + "," + str(j)
            if gui.Table_Target_Testing.item(count_add,j) != "":
                b +=1
            else:
                b +=0
                msg_Tar_Tes = msg_Tar_Tes + "," + str(j)
        if (count_add < D_Training):
            for i in range(0, (number_of_channel * 2) - 1):
                Kumpul_D_Training[count_add,i] = Hasil[i]
            if (a == 3 ):
                for j in range(3):
                    Kumpul_D_Target_Training[count_add,j] = float(gui.Table_Target_Training.item(count_add,j))
            else:
                msg = QMessageBox()
                Tampil_Train = "Isi Pada Baris ke - " + count_add + " Kolom ke - " + msg_Tar_Tra
                msg.setText(Tampil_Train)
                msg.exec_()
        elif ((count_add >= D_Training) and (count_add < D_Testing)):
            for i in range(0, (number_of_channel * 2) - 1):
                Kumpul_D_Training[count_add, i] = Hasil[i]
            if (b==3):
                for j in range(3):
                    Kumpul_D_Target_Testing[count_add, j] = float(gui.Table_Target_Testing.item(count_add, j))
            else:
                msg = QMessageBox()
                Tampil_Test = "Isi Pada Baris ke - " + count_add + " Kolom ke - " + msg_Tar_Tes
                msg.setText(Tampil_Test)
                msg.exec_()
    gui.AddData.clicked.connect(Add)
    def Del():
        global count
        count_del = count - 1
        if (count_del < D_Training):
            for i in range(0, (number_of_channel * 2) - 1):
                gui.Table_Training.setItem(count_del, i, QtGui.QTableWidgetItem(str("")))
            for j in range(3):
                gui.Table_Target_Testing.item(count,i,"")
        elif ((count_del >= D_Training) and (count_del < D_Testing)):
            for i in range(0, (number_of_channel * 2) - 1):
                gui.Table_Testing.setItem(count_del, i, QtGui.QTableWidgetItem(str("")))
        count = count_del
    gui.DeleteData.clicked.connect(Del)
    # Training
    def Train():
        global Kumpul_D_Testing, Kumpul_D_Training,Kumpul_D_Target_Testing,Kumpul_D_Target_Training,P,R
        if __name__ == "__main__":
            direct_save = gui.Directory_Line.text()
            error = round(float(gui.error.text()),6)
            print(error)
            print(direct_save)
            if direct_save!="":
                print("Mulai Train")
                Train = Training_Data(Kumpul_D_Training,
                                      Kumpul_D_Target_Training, Kumpul_D_Testing,
                                      Kumpul_D_Target_Testing, direct_save, error)
                R = float(Train.ELM_TD())
                hasil_error_training = "Error Training = " + str(float(round(P,5)))
                hasil_error_testing = "Error Testing = " + str(float(round(R,5)))
                gui.status_hasil_Training.setText(hasil_error_training)
                gui.status_hasil_Testing.setText(hasil_error_testing)
                print("Data Berhasil = ",round(R,5))
    gui.TrainingData.clicked.connect(Train)
    # Integrasi
    def Integra():
        if __name__ == "__main__":
            if Integrasi:
                port = gui.COM.currentText()
                dir = gui.Directory_Line.text()
                rte = Realtime(port=port,dir=dir)
                rte.main_process()
    gui.Run_Integrasi.clicked.connect(Integra)
    def Stop_Integra():
        if __name__ == "__main__":
            if Integrasi:
                port = gui.COM.currentText()
                B_erhenti = Cek_Robot()
                B_erhenti.Kirim_Arduino(3, port=str(port), baudrate=9600)
                Emotiv.quit()
    gui.Stop_Integrasi.clicked.connect(Stop_Integra)

    # Save Data Table
    def Save_Traning():
        global Kumpul_D_Training, Training
        if Training:
            path = str(QtGui.QFileDialog.getSaveFileName(None, 'Select a folder to Save File', "*.csv"))
            np.savetxt(path, Kumpul_D_Training, delimiter=" ")
    def Save_Testing():
        global Kumpul_D_Testing, Training
        if Training:
            path = str(QtGui.QFileDialog.getSaveFileName(None, 'Select a folder to Save File', "*.csv"))
            np.savetxt(path, Kumpul_D_Testing, delimiter=" ")
    def Save_Target_Training():
        global Kumpul_D_Target_Training, Training
        if Training:
            path = str(QtGui.QFileDialog.getSaveFileName(None, 'Select a folder to Save File', "*.csv"))
            np.savetxt(path, Kumpul_D_Target_Training, delimiter=" ")
    def Save_Target_Testing():
        global Kumpul_D_Target_Testing, Training
        if Training:
            path = str(QtGui.QFileDialog.getSaveFileName(None, 'Select a folder to Save File', "*.csv"))
            np.savetxt(path, Kumpul_D_Target_Testing, delimiter=" ")
    gui.save_training.clicked.connect(Save_Traning)
    gui.save_testing.clicked.connect(Save_Testing)
    gui.save_T_training.clicked.connect(Save_Target_Training)
    gui.save_T_testing.clicked.connect(Save_Target_Testing)
    # Load Data Table
    def Load_Training():
        global Kumpul_D_Testing, Kumpul_D_Training, Hasil, Kumpul_D_Target_Testing, Kumpul_D_Target_Training, Training
        if Training:
            path = str(QtGui.QFileDialog.getOpenFileName(None
                                                         , 'Select a folder to Open File'
                                                         ,"./Record Data"
                                                         ,"Open File (*.csv *.txt)"))
            Kumpul_D_Training = np.genfromtxt(fname=path, delimiter="")
            a,b = Kumpul_D_Training.shape
            gui.Table_Training.setRowCount(a)
            gui.Table_Training.setColumnCount(b)
            for i in range(a):
                for j in range(b):
                    gui.Table_Training.setItem(i, j, QtGui.QTableWidgetItem(str(Kumpul_D_Training[i,j])))
    def Load_Testing():
        global Kumpul_D_Testing, Kumpul_D_Training, Hasil, Kumpul_D_Target_Testing, Kumpul_D_Target_Training, Training
        if Training:
            path = str(QtGui.QFileDialog.getOpenFileName(None
                                                         , 'Select a folder to Open File'
                                                         ,"./Record Data"
                                                         ,"Open File (*.csv *.txt)"))
            Kumpul_D_Testing = np.genfromtxt(fname=path, delimiter="")
            a, b = Kumpul_D_Testing.shape
            gui.Table_Testing.setRowCount(a)
            gui.Table_Testing.setColumnCount(b)
            for i in range(a):
                for j in range(b):
                    gui.Table_Testing.setItem(i, j, QtGui.QTableWidgetItem(str(Kumpul_D_Testing[i,j])))
    def Load_Target_Training():
        global Kumpul_D_Testing, Kumpul_D_Training, Hasil, Kumpul_D_Target_Testing, Kumpul_D_Target_Training, Training
        if Training:
            path = str(QtGui.QFileDialog.getOpenFileName(None
                                                         , 'Select a folder to Open File'
                                                         ,"./Record Data"
                                                         ,"Open File (*.csv *.txt)"))
            Kumpul_D_Target_Training = np.genfromtxt(fname=path, delimiter="")
            a, b = Kumpul_D_Target_Training.shape
            gui.Table_Target_Training.setRowCount(a)
            gui.Table_Target_Training.setColumnCount(b)
            for i in range(a):
                for j in range(b):
                    gui.Table_Target_Training.setItem(i, j, QtGui.QTableWidgetItem(str(Kumpul_D_Target_Training[i,j])))
    def Load_Target_Testing():
        global Kumpul_D_Testing, Kumpul_D_Training, Hasil, Kumpul_D_Target_Testing, Kumpul_D_Target_Training, Training
        if Training:
            path = str(QtGui.QFileDialog.getOpenFileName(None
                                                         , 'Select a folder to Open File'
                                                         ,"./Record Data"
                                                         ,"Open File (*.csv *.txt)"))
            Kumpul_D_Target_Testing = np.genfromtxt(fname=path, delimiter="")
            a, b = Kumpul_D_Target_Testing.shape
            gui.Table_Target_Testing.setRowCount(a)
            gui.Table_Target_Testing.setColumnCount(b)
            for i in range(a):
                for j in range(b):
                    gui.Table_Target_Testing.setItem(i, j, QtGui.QTableWidgetItem(str(Kumpul_D_Target_Testing[i,j])))
    gui.Load_training.clicked.connect(Load_Training)
    gui.Load_testing.clicked.connect(Load_Testing)
    gui.Load_T_training.clicked.connect(Load_Target_Training)
    gui.Load_T_testing.clicked.connect(Load_Target_Testing)

    form.show()
    form.update()
    app.exec_()
    # rte = Realtime()
    # rte.main_process()