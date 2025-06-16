# Test of digital heterodyning
# python3

import sys
import math 

import numpy as np
from numpy.fft import fft, ifft, fftfreq

import matplotlib.pyplot as plt
import matplotlib.axes as ax

if __name__ == '__main__':
    
    #Channel A frequency and phase
    fa = 5000.0
    pa = 0.0 #Between 0 and 2pi
    #pa = math.pi/4.0
    
    #Channel B frequency and phase
    fb = 4900.0
    pb = 0.0 #Between 0 and 2pi
    
    #Sample Time (seconds)
    t=0.01
    
    SamplingRate = 4.0 #(Note the Nyquist limit is 2)
    
    Plots = True
    
    #Number of samples
    if (fa>fb):
        NSamples=int(SamplingRate*fa*t)+1
        #NSamples=int(SamplingRate*fa*t)
    else:
        NSamples=int(SamplingRate*fb*t)+1
        #NSamples=int(SamplingRate*fb*t)
        
    print("\nNumber of Samples: ", NSamples)
    
    #Create Arrays for Channels A and B
    ArrA=np.zeros(NSamples)
    ArrB=np.zeros(NSamples)
    ArrTime = np.linspace(0,t,NSamples)
    ArrFreq=np.zeros(NSamples)
    
    ArrCos = np.zeros(NSamples)
    ArrSin = np.zeros(NSamples)
    
    print("\nArrA: \n", ArrA)
    print("\nArrB: \n", ArrB)
    
    print("\nArrTime: \n", ArrTime)
    
    #Set Array Values including the phases
    for index, x in np.ndenumerate(ArrA):
        ArrA[index]=math.sin(2.0*math.pi*fa*ArrTime[index]+pa)
    
    print("\nArrA: \n", ArrA)
    
    for index, x in np.ndenumerate(ArrA):
        ArrB[index]=math.sin(2.0*math.pi*fb*ArrTime[index]+pb)
    
    print("\nArrB: \n", ArrB)
    
    #Heterodyne the two signals
    for index, x in np.ndenumerate(ArrA):
        ArrCos[index]=math.cos(2.0*math.pi*fb*ArrTime[index])*(ArrA[index])
        ArrSin[index]=math.sin(2.0*math.pi*fb*ArrTime[index])*(ArrA[index])
    
    if Plots == True:
        #Plot Channel A 
        figure=plt.figure(1, figsize=(10, 6))
        #Generate the axes
        #This parameter is the dimensions [left, bottom, width, height] of the new axes.  
        ax1 = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        ax1.plot(ArrTime, ArrA, marker='.', linestyle='', markersize=1, color='blue')
        plt.show(block=False)
        
        #Plot Channel B
        figure=plt.figure(2, figsize=(10, 6))
        #Generate the axes
        #This parameter is the dimensions [left, bottom, width, height] of the new axes.  
        ax1 = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        ax1.plot(ArrTime, ArrB, marker='.', linestyle='', markersize=1, color='green')
        plt.show(block=False)
        
        #Plot Heterodyne
        figure=plt.figure(3, figsize=(10, 6))
        #Generate the axes
        #This parameter is the dimensions [left, bottom, width, height] of the new axes.  
        ax1 = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        ax1.plot(ArrTime, ArrCos, marker='.', linestyle='', markersize=1, color='red')
        
        
        figure=plt.figure(4, figsize=(10, 6))
        #Generate the axes
        #This parameter is the dimensions [left, bottom, width, height] of the new axes.  
        ax1 = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        ax1.plot(ArrTime, ArrSin, marker='.', linestyle='', markersize=1, color='black')
        plt.show(block=False)
    
    
    #Generate the FFT
    X = fft(ArrCos)
    
    #We now need to find the inverse of the time...
    for index, x in np.ndenumerate(ArrTime):
        ArrFreq[index]=1.0/ArrTime[index]
        
    #Now find the frequency bins:
    n=ArrCos.size
    timestep=ArrTime[1]
    freq=fftfreq(n,d=timestep)
    
    print("\nfreq: \n", freq)
    
    #N = len(Arrcos)
    #n = np.arange(N) #Returns evenly spaced values between an interval
    #T = N/SamplingRate
    #freq = n/T 
    #print("\nfreq: ", freq, len(freq))

    #Plot the fft

    figure=plt.figure(5, figsize=(10, 6))
    plt.subplot(121)

    plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 2*fa)

    plt.subplot(122)
    plt.plot(ArrTime, ifft(X), marker='.', linestyle='', markersize=1, color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    
    plt.show(block=False)

################################################################################

    #Script End
    #endTime=datetime.now()
    #timeTaken=endTime-startTime
    
    #print("\nScript Run Time: ", timeTaken, "\n")
    
    input("\nPress Enter to continue...")
    print("fin")
    
    sys.exit()
    
################################################################################ 