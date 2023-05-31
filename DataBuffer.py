import math
import numpy as np

class DataBuffer:
    def __init__(self, data_dir,  data_number, dtype=int, block_size = 1000, buffer_number=2, file_name="blk_", max_seq_len=None):
        self.block_size = block_size
        self.block_number = math.ceil(data_number / block_size)
        self.block_status = [False]*self.block_number # true if blk i in buffer
        self.block_access_counts = [0]*self.block_number # access count of blk i
        self.buffer_number = buffer_number 
        self.data_dir = data_dir
        self.dtype = dtype
        self.file_name = file_name
        #self.data_buffer = [ np.empty((*self.dim), , dtype=self.dtype) ] * self.block_size #the data buffer
        self.data_buffer_map = {} #key: blick id, value: data block in the buffer
        self.max_seq_len = max_seq_len

    def initialize(self):
        self.block_access_counts = [0]*self.block_number # access count of blk i
        
    def __get_block_id(self, dataID):
        return dataID // self.block_size

    def __get_data_idx(self, dataID):
        return dataID % self.block_size

    def __check_exist(self, dataID):
        blk_id = self.__get_block_id( dataID )
        return self.block_status[blk_id]
    
    def __read_block(self, block_id):
        data_block =  None # [ np.empty((*self.dim), , dtype=self.dtype) ] * self.block_size
        if self.data_dir.endswith("\\"):
            data_block= list( np.load(self.data_dir + self.file_name + str(block_id) + '.npy') )
        else:
            data_block= list( np.load(self.data_dir + "\\"+ self.filename + str(block_id) + '.npy') )

        return data_block

    def __replace_data_buffer(self, block_id, data_block):
        if len(self.data_buffer_map)>0:
            #print("self.data_buffer_map.keys():", list(self.data_buffer_map.keys()))
            counts = { blk_id:self.block_access_counts[blk_id] for blk_id in 
            self.data_buffer_map.keys()}
            #print("counts: ", counts)
            max_access_block_id = max(counts.items(), key=lambda item: item[1])[0]
            #print("max_access_block_id:", max_access_block_id)
            del self.data_buffer_map[max_access_block_id]
            self.block_status[max_access_block_id] = False
            #print("block ", max_access_block_id , " is replaced by blk ",block_id )
        self.data_buffer_map[block_id] = data_block
        self.block_status[block_id] = True

        
    def get_data(self, dataID):
        blk_id = self.__get_block_id( dataID )
        data_block=None
        if self.__check_exist(dataID): #exist in buffer
            data_block = self.data_buffer_map[blk_id]
        else: # not in the buffer, read block in the disk
            data_block = self.__read_block(blk_id)
            self. __replace_data_buffer(blk_id, data_block)

        self.block_access_counts[blk_id] += 1
        #print("data_block[self.__get_data_idx(dataID)]:", data_block[self.__get_data_idx(dataID)])
        if self.max_seq_len is None:
            return data_block[self.__get_data_idx(dataID)]
        else:
            return data_block[self.__get_data_idx(dataID)][0:self.max_seq_len]
            
            
            
        
        
