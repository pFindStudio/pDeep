import os
import struct

def get_scan_charge_by_specname(specname): #pParse format
    items = specname.split(".")
    return int(items[-4]), int(items[-3]) #scan, charge
    
def get_raw_name(filename):
    filename = os.path.basename(filename)
    if filename.rfind("_") != -1:
        return filename[:filename.rfind("_")]
    else:
        return filename[:filename.rfind(".")]
        
def GetReader(filename):
    if filename.endswith(".pf2"):
        reader = PF2Reader()
    elif filename.endswith(".mgf"):
        reader = MGFReader()
    elif filename.endswith(".ms2"):
        reader = MS2Reader()
    else:
        return None
    reader.open_and_index_file(filename)
    return reader

class PF2Reader:
    def __init__(self):
        self.file = None
        
    def open_and_index_file(self, filename):
        self.file = open(filename, 'rb')
        
        self.raw_name = get_raw_name(filename)
        
        self.scanidx = {}
        f = open(filename+'idx','rb')
        while True:
            chunk = f.read(8)
            if not chunk: break
            scan_no, index = struct.unpack('2i',chunk)
            self.scanidx[scan_no] = index
        f.close()
    
    def read_a_peaklist(self, scan):
        # only read the (mz, inten) list
        self.file.seek(self.scanidx[scan])
        scan, nPeak = struct.unpack("2i",self.file.read(8))
        mz_int = struct.unpack(str(nPeak*2)+"d", self.file.read(nPeak*2*8))
        peaklist = []
        for i in range(nPeak):
            mz = mz_int[i*2]
            inten = mz_int[i*2+1]
            peaklist.append( (mz, inten) )
        return peaklist
        
    def close(self):
        if self.file is not None: self.file.close()
        
    def __del__(self):
        self.close()
        
class MS2Reader:
    def __init__(self):
        self.file = None
        
    def open_and_index_file(self, filename):
        self.file = open(filename)
        
        self.raw_name = get_raw_name(filename)
        
        self.scanidx = {}
        while True:
            line = self.file.readline()
            if line == "": break
            elif line.startswith("S"):
                idx = self.file.tell()
                scan = int(line.split("\t")[1])
                self.scanidx[scan] = idx
    
    def read_a_peaklist(self, scan):
        # only read the (mz, inten) list
        self.file.seek(self.scanidx[scan])
        peaklist = []
        while True:
            line = f.readline()
            if line == "": break
            elif line.startswith("S"): break
            elif line[0].isdigit():
                mz, inten = line.strip().split()
                mz, inten = float(mz), float(inten)
                peaklist.append( (mz, inten) )
        return peaklist
        
    def close(self):
        if self.file is not None: self.file.close()
        
    def __del__(self):
        self.close()
        
class MGFReader:
    def __init__(self):
        self.file = None
        
    def open_and_index_file(self, filename):
        self.file = open(filename)
        
        self.raw_name = get_raw_name(filename)
        
        self.scanidx = {}
        while True:
            line = self.file.readline()
            if line == "": break
            elif line.startswith("BEGIN IONS"):
                idx = self.file.tell()
            elif line.startswith("TITLE"):
                scan,_ = get_scan_charge_by_specname(line.strip())
                self.scanidx[scan] = idx
    
    def read_a_peaklist(self, scan):
        # only read the (mz, inten) list
        self.file.seek(self.scanidx[scan])
        peaklist = []
        while True:
            line = f.readline()
            if line == "": break
            elif line.startswith("END IONS"): break
            elif line[0].isdigit():
                mz, inten = line.strip().split()
                mz, inten = float(mz), float(inten)
                peaklist.append( (mz, inten) )
        return peaklist
        
    def close(self):
        if self.file is not None: self.file.close()
        
    def __del__(self):
        self.close()