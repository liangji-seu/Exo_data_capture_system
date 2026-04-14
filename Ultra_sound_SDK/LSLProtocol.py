from pylsl import StreamInfo, StreamOutlet, local_clock

class SendSLStream:
    def __init__(self,type,channelsNum,srate):
        self._info = StreamInfo('ElonxiLSL',type,channelsNum,srate,'float32','ElonxiLSLID3849487')
        self._outlet = StreamOutlet(self._info)

    def sendData(self,sample):
        for item in sample:
            self._outlet.push_sample(item)


class SendSLSMarkers:
    def __init__(self):
        self._info = StreamInfo('ElonxiLSL','Markers',1,0,'string','ElonxiLSLID3849489')
        self._outlet = StreamOutlet(self._info)

    def sendData(self,markernames):
        self._outlet.push_sample([markernames])