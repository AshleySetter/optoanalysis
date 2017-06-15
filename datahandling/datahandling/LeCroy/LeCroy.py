import warnings

class HDO6104:
        """
        Class for communicating with the Teledyne LeCroy Oscilloscope.
        """
        def __init__(self, address='152.78.194.16'):
                """
                Initialises the connection to the Oscilloscope.

                Parameters
                ----------
                address : string
                    IP address of the oscilloscope.
                """
                self.address = address
                import vxi11
                self.connection = vxi11.Instrument(address)

                self.write = self.connection.write
                self.read  = self.connection.read
                self.ask   = self.connection.ask
                self.read_raw = self.connection.read_raw

        def raw(self, channel=1):
                """
                Reads the raw input from the oscilloscope.

                Parameters
                ----------
                channel : int
                    channel number of read

                Returns
                -------
                rawData : bytes
                    raw binary data read from the oscilloscope
                """
                self.waitOPC()
                self.write('COMM_FORMAT DEF9,WORD,BIN')
                self.write('C%u:WAVEFORM?' % channel)
                return self.read_raw()
        
        def data(self, channel=1):
                """
                Reads the raw input from the scope and interprets it
                returning the header information, time, voltage and 
                raw integers read with the ADC.

                Parameters
                ----------
                channel : int
                    channel number of read

                Returns
                -------
                WAVEDESC : dict
                    dictionary containing some properties of the time trace and oscilloscope
                    settings extracted from the header file.
                x : ndarray
                    The array of time values recorded by the oscilloscope
                y : ndarray
                    The array of voltage values recorded by the oscilloscope
                integers : ndarray
                    The array of raw integers recorded from the ADC and stored in the binary file
                """
                raw = self.raw(channel) # Grab waveform from scope
                return InterpretWaveform(raw)

        def waitOPC(self):
                """
                Waits for a response from the oscilloscope indicating that
                processing is complete and it is ready to receive more commands.
                Function sleeps until the oscilloscope is ready.
                """
                from time import sleep
                self.write('WAIT')
                while not self.opc():
                        sleep(1)
                
        def opc(self):
                """
                Asks the oscilloscope if it is done processing data.

                Returns
                -------
                IsDoneProcessing : bool
                    returns False if oscilloscope is still busy, 
                    True is oscilloscope is done processing last commands.
                """
                return self.ask('*OPC?')[-1] == '1'


def InterpretWaveform(raw, integersOnly=False, headersOnly=False):
        """
        Take the raw binary from a file saved from the LeCroy, read from a file using 
        the 2 lines:
        with open(filename, "rb") as file:
            raw = file.read()
        And extracts various properties of the saved time trace.
        
        Parameters
        ----------
        raw : bytes
            Bytes object containing the binary contents of the saved raw/trc file
        integersOnly : bool, optional
            If True, only returns the unprocessed integers (read from the ADC) 
            rather than the signal in volts. Defaults to False. 
        headersOnly : bool, optional
            If True, only returns the file header. Defaults to False. 

        Returns
        -------
        WAVEDESC : dict
            dictionary containing some properties of the time trace and oscilloscope
            settings extracted from the header file.
        x : ndarray
            The array of time values recorded by the oscilloscope
        y : ndarray
            The array of voltage values recorded by the oscilloscope
        integers : ndarray
            The array of raw integers recorded from the ADC and stored in the binary file
        """
        MissingData = False
        from struct import unpack
        
        if raw[0:1] != b'#':
                cmd = raw.split(b',')[0]  # "C1:WF ALL" or similar
                wave = raw[len(cmd)+1:]   # Remove the above command text (and trailing
        else:
                wave = raw

        del raw

        if wave[0:1] != b'#':
                warnings.warn('Waveform format not as expected, time trace may be missing data')
                MissingData = True
                
        n = int(wave[1:2])          # number of digits in length of data
        N = int(wave[2:2+n])      # number describing length of data

        if wave.endswith(b'\n'):
                wave = wave[:-1]

        wave = wave[2+n:]

        if N != len(wave):
                warnings.warn('Length of waveform not as expected, time trace may be missing data')

        # Code to parse WAVEDESC generated by parsing template, returned from scope query "TEMPLATE?"
        # Note that this is not well tested and will not handle unusual settings
        WAVEDESC = dict()
        WAVEDESC['DESCRIPTOR_NAME'] = wave[0:16].strip(b'\x00')
        WAVEDESC['TEMPLATE_NAME'] = wave[16:32].strip(b'\x00')
        WAVEDESC['COMM_TYPE'] = {0: 'byte',1: 'word'}[unpack(b"<H", wave[32:34])[0]]
        WAVEDESC['COMM_ORDER'] = {0: 'HIFIRST',1: 'LOFIRST'}[unpack("<H", wave[34:36])[0]]
        WAVEDESC['WAVE_DESCRIPTOR'] = unpack('<l', wave[36:40])[0]
        WAVEDESC['USER_TEXT'] = unpack('<l', wave[40:44])[0]
        WAVEDESC['RES_DESC1'] = unpack('<l', wave[44:48])[0]
        WAVEDESC['TRIGTIME_ARRAY'] = unpack('<l', wave[48:52])[0]
        WAVEDESC['RIS_TIME_ARRAY'] = unpack('<l', wave[52:56])[0]
        WAVEDESC['RES_ARRAY1'] = unpack('<l', wave[56:60])[0]
        WAVEDESC['WAVE_ARRAY_1'] = unpack('<l', wave[60:64])[0]
        WAVEDESC['WAVE_ARRAY_2'] = unpack('<l', wave[64:68])[0]
        WAVEDESC['RES_ARRAY2'] = unpack('<l', wave[68:72])[0]
        WAVEDESC['RES_ARRAY3'] = unpack('<l', wave[72:76])[0]
        WAVEDESC['INSTRUMENT_NAME'] = wave[76:92].strip(b'\x00')
        WAVEDESC['INSTRUMENT_NUMBER'] = unpack('<l', wave[92:96])[0]
        WAVEDESC['TRACE_LABEL'] = wave[96:112].strip(b'\x00')
        WAVEDESC['RESERVED1'] = unpack('<h', wave[112:114])[0]
        WAVEDESC['RESERVED2'] = unpack('<h', wave[114:116])[0]
        WAVEDESC['WAVE_ARRAY_COUNT'] = unpack('<l', wave[116:120])[0]
        WAVEDESC['PNTS_PER_SCREEN'] = unpack('<l', wave[120:124])[0]
        WAVEDESC['FIRST_VALID_PNT'] = unpack('<l', wave[124:128])[0]
        WAVEDESC['LAST_VALID_PNT'] = unpack('<l', wave[128:132])[0]
        WAVEDESC['FIRST_POINT'] = unpack('<l', wave[132:136])[0]
        WAVEDESC['SPARSING_FACTOR'] = unpack('<l', wave[136:140])[0]
        WAVEDESC['SEGMENT_INDEX'] = unpack('<l', wave[140:144])[0]
        WAVEDESC['SUBARRAY_COUNT'] = unpack('<l', wave[144:148])[0]
        WAVEDESC['SWEEPS_PER_ACQ'] = unpack('<l', wave[148:152])[0]
        WAVEDESC['POINTS_PER_PAIR'] = unpack('<h', wave[152:154])[0]
        WAVEDESC['PAIR_OFFSET'] = unpack('<h', wave[154:156])[0]
        WAVEDESC['VERTICAL_GAIN'] = unpack('<f', wave[156:160])[0]
        WAVEDESC['VERTICAL_OFFSET'] = unpack('<f', wave[160:164])[0]
        WAVEDESC['MAX_VALUE'] = unpack('<f', wave[164:168])[0]
        WAVEDESC['MIN_VALUE'] = unpack('<f', wave[168:172])[0]
        WAVEDESC['NOMINAL_BITS'] = unpack('<h', wave[172:174])[0]
        WAVEDESC['NOM_SUBARRAY_COUNT'] = unpack('<h', wave[174:176])[0]
        WAVEDESC['HORIZ_INTERVAL'] = unpack('<f', wave[176:180])[0]
        WAVEDESC['HORIZ_OFFSET'] = unpack('<d', wave[180:188])[0]
        WAVEDESC['PIXEL_OFFSET'] = unpack('<d', wave[188:196])[0]
        WAVEDESC['VERTUNIT'] = wave[196:244].strip(b'\x00')
        WAVEDESC['HORUNIT'] = wave[244:292].strip(b'\x00')
        WAVEDESC['HORIZ_UNCERTAINTY'] = unpack('<f', wave[292:296])[0]
        WAVEDESC['TRIGGER_TIME'] = wave[296:312] # Format time_stamp not implemented
        WAVEDESC['ACQ_DURATION'] = unpack('<f', wave[312:316])[0]
        WAVEDESC['RECORD_TYPE'] = {0: 'single_sweep',1: 'interleaved',2: 'histogram',3: 'graph',4: 'filter_coefficient',5: 'complex',6: 'extrema',7: 'sequence_obsolete',8: 'centered_RIS',9: 'peak_detect'}[unpack("<H", wave[316:318])[0]]
        WAVEDESC['PROCESSING_DONE'] = {0: 'no_processing',1: 'fir_filter',2: 'interpolated',3: 'sparsed',4: 'autoscaled',5: 'no_result',6: 'rolling',7: 'cumulative'}[unpack("<H", wave[318:320])[0]]
        WAVEDESC['RESERVED5'] = unpack('<h', wave[320:322])[0]
        WAVEDESC['RIS_SWEEPS'] = unpack('<h', wave[322:324])[0]
        WAVEDESC['TIMEBASE'] = {0: '1_ps/div',1: '2_ps/div',2: '5_ps/div',3: '10_ps/div',4: '20_ps/div',5: '50_ps/div',6: '100_ps/div',7: '200_ps/div',8: '500_ps/div',9: '1_ns/div',10: '2_ns/div',11: '5_ns/div',12: '10_ns/div',13: '20_ns/div',14: '50_ns/div',15: '100_ns/div',16: '200_ns/div',17: '500_ns/div',18: '1_us/div',19: '2_us/div',20: '5_us/div',21: '10_us/div',22: '20_us/div',23: '50_us/div',24: '100_us/div',25: '200_us/div',26: '500_us/div',27: '1_ms/div',28: '2_ms/div',29: '5_ms/div',30: '10_ms/div',31: '20_ms/div',32: '50_ms/div',33: '100_ms/div',34: '200_ms/div',35: '500_ms/div',36: '1_s/div',37: '2_s/div',38: '5_s/div',39: '10_s/div',40: '20_s/div',41: '50_s/div',42: '100_s/div',43: '200_s/div',44: '500_s/div',45: '1_ks/div',46: '2_ks/div',47: '5_ks/div',100: 'EXTERNAL'}[unpack("<H", wave[324:326])[0]]
        WAVEDESC['VERT_COUPLING'] = {0: 'DC_50_Ohms',1: 'ground',2: 'DC_1MOhm',3: 'ground',4: 'AC_1MOhm'}[unpack("<H", wave[326:328])[0]]
        WAVEDESC['PROBE_ATT'] = unpack('<f', wave[328:332])[0]
        WAVEDESC['FIXED_VERT_GAIN'] = {0: '1_uV/div',1: '2_uV/div',2: '5_uV/div',3: '10_uV/div',4: '20_uV/div',5: '50_uV/div',6: '100_uV/div',7: '200_uV/div',8: '500_uV/div',9: '1_mV/div',10: '2_mV/div',11: '5_mV/div',12: '10_mV/div',13: '20_mV/div',14: '50_mV/div',15: '100_mV/div',16: '200_mV/div',17: '500_mV/div',18: '1_V/div',19: '2_V/div',20: '5_V/div',21: '10_V/div',22: '20_V/div',23: '50_V/div',24: '100_V/div',25: '200_V/div',26: '500_V/div',27: '1_kV/div'}[unpack("<H", wave[332:334])[0]]
        WAVEDESC['BANDWIDTH_LIMIT'] = {0: 'off',1: 'on'}[unpack("<H", wave[334:336])[0]]
        WAVEDESC['VERTICAL_VERNIER'] = unpack('<f', wave[336:340])[0]
        WAVEDESC['ACQ_VERT_OFFSET'] = unpack('<f', wave[340:344])[0]
        WAVEDESC['WAVE_SOURCE'] = {0: 'CHANNEL_1',1: 'CHANNEL_2',2: 'CHANNEL_3',3: 'CHANNEL_4',9: 'UNKNOWN'}[unpack("<H", wave[344:346])[0]]

        if len(wave[346:]) != WAVEDESC['WAVE_ARRAY_1']:
                warnings.warn('Binary data not the expected length, time trace may be missing data')
                MissingData = True

        if headersOnly:
                return WAVEDESC
        else:
                from numpy import fromstring, int16, arange
                if MissingData != True:
                        integers = fromstring(wave[346:], dtype=int16)
                else:
                        integers = fromstring(wave[346:][:-1], dtype=int16)
                        
                if integersOnly:
                        return (WAVEDESC, integers)
                else:
                        y = integers * WAVEDESC['VERTICAL_GAIN'] - WAVEDESC['VERTICAL_OFFSET'] 
                        x = arange(len(integers)) * WAVEDESC['HORIZ_INTERVAL'] + WAVEDESC['HORIZ_OFFSET']
                        return (WAVEDESC, x, y, integers)

