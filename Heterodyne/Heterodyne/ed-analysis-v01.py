import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sys

# NOTE - this is about 1/2 the size of the least significant bit in the ADC, which is 1mV.
# You can clearly see this in the upper figure - the horizontal stripes in the measured 
# voltage are separated by 0.001V. This is a strong indicator that the noise floor in the
# amplitude spectral density is due to quantisation in the ADC.

if __name__ == '__main__':

    print("")

    plotData=False
    plotWelch=True

    # open the first file and read in all the data, into a list of ascii elements
    with open('Agilent_33250A_80mVpp_2023_04_10b.csv') as f1:
        lines=f1.readlines()
        
    # figure out the number of data lines
    linelength=len(lines)
    print("Number of Lines in the Data File: ", linelength)

    # allocate memory to a data array
    data1=np.zeros(linelength)

    # cast the ascii data into floats entry by entry
    elcount=0;
    for line in lines:
        data1[elcount]=float(line)
        elcount+=1

    #Calculate the RMS of the data values
    rms=np.sqrt(np.mean(data1**2))

    print("RMS: ", rms)
    print("")


    # sampling rate extracted from the John Gallop notebook, final page, where the file path
    # indicates a file with the same name as supplied by John as data.
    samplerate=1e6

    # make a plot of the data as it appears in the file
    if (plotData==True):
        fig = plt.figure(1, figsize=(10, 6))
        ax = plt.axes()

        time = np.linspace(0,(linelength-1)/samplerate, linelength)
        ax.plot(time,data1,'.k')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('voltage (V)')
        ax.grid(True)
        plt.show(block=False)
    
    #x: array_like
        #Time series of measurement values

    #fsfloat, optional
        #Sampling frequency of the x time series. Defaults to 1.0.

    #windowstr or tuple or array_like, optional
        #Desired window to use. If window is a string or tuple, it is passed to get_window to generate the window values, which are DFT-even by default. See get_window for a list of windows and required parameters. If window is array_like it will be used directly as the window and its length must be nperseg. Defaults to a Hann window.
    
    #npersegint, optional
        #Length of each segment. Defaults to None, but if window is str or tuple, is set to 256, and if window is array_like, is set to the length of the window.

    #noverlapint, optional
        #Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.

    #nfftint, optional
        #Length of the FFT used, if a zero padded FFT is desired. If None, the FFT length is nperseg. Defaults to None.

    #detrendstr or function or False, optional
        #Specifies how to detrend each segment. If detrend is a string, it is passed as the type argument to the detrend function. If it is a function, it takes a segment and returns a detrended segment. If detrend is False, no detrending is done. Defaults to ‘constant’.

    #return_onesidedbool, optional
        #If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Defaults to True, but for complex data, a two-sided spectrum is always returned.

    #scaling{ ‘density’, ‘spectrum’ }, optional
        #Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz and computing the squared magnitude spectrum (‘spectrum’) where Pxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density’

    #axisint, optional
        #Axis along which the periodogram is computed; the default is over the last axis (i.e. axis=-1).

    #average{ ‘mean’, ‘median’ }, optional
        #Method to use when averaging periodograms. Defaults to ‘mean’.
    
    
    # make welch estimate with no windowing
    f,wel = signal.welch(
                        data1,fs=samplerate,window='boxcar',nperseg=32768, 
                        noverlap=0, nfft=None, 
                        detrend=False, return_onesided=True, scaling='density',
                        axis=-1, average='mean')
                        
    # make welch estimate with no windowing
    f2,wel2 = signal.welch(
                        data1,fs=samplerate,window='hann',nperseg=16384, 
                        noverlap=8192, nfft=None, 
                        detrend=False, return_onesided=True, scaling='density',
                        axis=-1, average='mean') 
                      
    #Returns:

    #f:ndarray
        #Array of sample frequencies.
    
    #Pxx:ndarray
        #Power spectral density or power spectrum of x.

                   
    if (plotWelch==True):
        fig2 = plt.figure(2, figsize=(8, 5))
        ax2=plt.axes()
        ax2.plot(f/1000,wel,'-k')
        ax2.set_yscale("log")
        ax2.set_xlabel('frequency (kHz)')
        ax2.set_ylabel('power spectral density (V^2/Hz)')
        ax2.grid(True)
        plt.show(block=False)

        fig3 = plt.figure(3, figsize=(8, 5))
        ax3=plt.axes()
        ax3.plot(f2/1000,wel2,'-k')
        ax3.set_yscale("log")
        ax3.set_xlabel('frequency (kHz)')
        ax3.set_ylabel('power spectral density (V^2/Hz)')
        ax3.grid(True)
        plt.show(block=False)

    # obtain resolution bandwidth
    resbw=f[1]-f[0]
    print('Resolution bandwidth is ', resbw)
    
    resbw2=f2[1]-f2[0]
    print('Resolution bandwidth 2 is ', resbw2)

    print("")

    # obtain spectral density of background noise by averaging over bins between 300 and 400 kHz
    nbins=len(f)
    print('Number of bins(1): ',nbins)
    
    nbins2=len(f2)
    print('Number of bins(2): ',nbins2)
    
    print("")
    
    firstbin=int(np.floor((300/500)*nbins))
    lastbin=int(np.floor((400/500)*nbins))
    noisefloor=np.mean(wel[firstbin:lastbin])
    
    firstbin2=int(np.floor((300/500)*nbins2))
    lastbin2=int(np.floor((400/500)*nbins2))
    noisefloor2=np.mean(wel2[firstbin2:lastbin2])
    
    print('Noise floor in first power spectral density is at ', noisefloor, 'V^2/Hz')
    print('Noise floor in second power spectral density is at ', noisefloor2, 'V^2/Hz')

    print("")

    # calculate corresponing RMS voltage fluctuation
    vfluct=np.sqrt(noisefloor*nbins*resbw)
    print('Noise floor corresponds to a voltage fluctuation of ', (vfluct*1000.0), 'millivolts.')
    
    # calculate corresponing RMS voltage fluctuation
    vfluct2=np.sqrt(noisefloor2*nbins2*resbw2)
    print('Noise floor corresponds to a voltage fluctuation of ', (vfluct2*1000.0), 'millivolts.')

   
    
################################################################################

    #Script End
    #endTime=datetime.now()
    #timeTaken=endTime-startTime
    
    #print("\nScript Run Time: ", timeTaken, "\n")
    
    input("\nPress Enter to continue...")
    print("fin")
    
    sys.exit()
    
################################################################################ 