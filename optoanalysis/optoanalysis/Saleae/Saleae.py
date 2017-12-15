import struct
import numpy as _np

def get_chunks(Array, Chunksize):
    """Generator that yields chunks of size ChunkSize"""
    for i in range(0, len(Array), Chunksize):
        yield Array[i:i + Chunksize]

def read_data_from_bin_file(fileName):
    """
    Loads the binary data stored in the a binary file and extracts the 
    data for each channel that was saved, along with the sample rate and length
    of the data array.

    Parameters
    ----------
    fileContent : bytes 
        bytes object containing the data from a .bin file exported from
        the saleae data logger.
    
    Returns
    -------
    ChannelData : list
        List containing a list which contains the data from each channel
    LenOf1Channel : int
        The length of the data in each channel
    NumOfChannels : int
        The number of channels saved
    SampleTime : float
        The time between samples (in seconds)
    SampleRate : float
        The sample rate (in Hz)
    """
    with open(fileName, mode='rb') as file: # b is important -> binary
        fileContent = file.read()

    (ChannelData, LenOf1Channel,
     NumOfChannels, SampleTime) = read_data_from_bytes(fileContent)
    
    return ChannelData, LenOf1Channel, NumOfChannels, SampleTime

def read_data_from_bytes(fileContent):
    """
    Takes the binary data stored in the binary string provided and extracts the 
    data for each channel that was saved, along with the sample rate and length
    of the data array.

    Parameters
    ----------
    fileContent : bytes 
        bytes object containing the data from a .bin file exported from
        the saleae data logger.
    
    Returns
    -------
    ChannelData : list
        List containing a list which contains the data from each channel
    LenOf1Channel : int
        The length of the data in each channel
    NumOfChannels : int
        The number of channels saved
    SampleTime : float
        The time between samples (in seconds)
    SampleRate : float
        The sample rate (in Hz)
    """
    TotalDataLen = struct.unpack('Q', fileContent[:8])[0] # Unsigned long long 
    NumOfChannels = struct.unpack('I', fileContent[8:12])[0] # unsigned Long
    SampleTime = struct.unpack('d', fileContent[12:20])[0]

    AllChannelData = struct.unpack("f" * ((len(fileContent) -20) // 4), fileContent[20:])
    #  ignore the heading bytes (= 20)
    # The remaining part forms the body, to know the number of bytes in the body do an integer division by 4 (since 4 bytes = 32 bits = sizeof(float)

    LenOf1Channel = int(TotalDataLen/NumOfChannels)

    ChannelData = list(get_chunks(AllChannelData, LenOf1Channel))
    
    return ChannelData, LenOf1Channel, NumOfChannels, SampleTime


def interpret_waveform(fileContent, RelativeChannelNo):
    """
    Extracts the data for just 1 channel and computes the corresponding
    time array (in seconds) starting from 0.
    
    Important Note: RelativeChannelNo is NOT the channel number on the Saleae data logger 
    it is the relative number of the channel that was saved. E.g. if you 
    save channels 3, 7 and 10, the corresponding RelativeChannelNos would
    be 0, 1 and 2.
    
    Parameters
    ----------
    fileContent : bytes 
        bytes object containing the data from a .bin file exported from
        the saleae data logger.
    RelativeChannelNo : int
        The relative order/position of the channel number in the saved
        binary file. See Important Note above!

    Returns
    -------    
    time : ndarray
        A generated time array corresponding to the data list
    Data : list
        The data from the relative channel requested
    SampleTime : float
        The time between samples (in seconds)
    """
    (ChannelData, LenOf1Channel,
     NumOfChannels, SampleTime) = read_data_from_bytes(fileContent)

    if RelativeChannelNo > NumOfChannels-1:
        raise ValueError("There are {} channels saved, you attempted to read relative channel number {}. Pick a relative channel number between {} and {}".format(NumOfChannels, RelativeChannelNo, 0, NumOfChannels-1))
    
    data = ChannelData[RelativeChannelNo]

    del(ChannelData)

    time = _np.arange(0, SampleTime*LenOf1Channel, SampleTime)

    return (0,SampleTime*LenOf1Channel,SampleTime), data
