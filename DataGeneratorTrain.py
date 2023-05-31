import numpy as np
import keras
import DataBuffer as db

#for text generation
#input: list_IDs=[encoder_input_ids, decoder_input_ids]
#output: [out1, out2]
#out1: [e1, e2, ..., e36], ei is either 0 or 1
#out2: [ tvector1, tvector2, .......], tvector_i: one-hot-encodeing [000,1,000](Max_Seq_Length)

class DataGeneratorTrain(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input_databuffer_params,output_databuffer_params, list_IDs,  batch_size=64, total_error_types=36, max_text_len=999, shuffle=True):
        'Initialization'
        self.total_error_types = total_error_types
        self.max_text_len = max_text_len
        self.batch_size = batch_size
        #self.err_labels = err_labels # out 1
        #self.messages = messages #out 2		
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.input_databuffer_params = input_databuffer_params        
        self.output_databuffer_params = output_databuffer_params
        # Generate input data buffer
        data_path = self.input_databuffer_params["data_path"]
        data_number = self.input_databuffer_params["data_number"]
        data_type = self.input_databuffer_params["data_type"]
        block_size = self.input_databuffer_params["block_size"]
        self.dbx1 = db.DataBuffer(data_path[0], data_number[0], data_type[0], block_size[0], file_name="x_train[0]_", max_seq_len=max_text_len )
        self.dbx2 = db.DataBuffer(data_path[1], data_number[1], data_type[1], block_size[1], file_name="x_train[1]_", max_seq_len=max_text_len )
        # Generate output data buffer
        data_path = self.output_databuffer_params["data_path"]
        data_number = self.output_databuffer_params["data_number"]
        data_type = self.output_databuffer_params["data_type"]
        block_size = self.output_databuffer_params["block_size"]
        self.dby1 = db.DataBuffer(data_path[0], data_number[0], data_type[0], block_size[0], file_name="y_train[0]_" )
        self.dby2 = db.DataBuffer(data_path[1], data_number[1], data_type[1], block_size[1], file_name="y_train[1]_", max_seq_len=max_text_len )
 
        self.on_epoch_end()
 
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs[0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[0][k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs[0]))
        self.dbx1.initialize()
        self.dbx2.initialize()
        self.dby1.initialize()
        self.dby2.initialize()
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization		
        X1  = np.empty((self.batch_size, self.max_text_len ), dtype=int)#encoder input
        X2  = np.empty((self.batch_size, self.max_text_len ), dtype=int)#decoder input        
        y1 = np.empty((self.batch_size, self.total_error_types), dtype=int)
        y2 = np.empty((self.batch_size, self.max_text_len, 1 ), dtype=int)
      
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X1[i,] = self.dbx1.get_data(ID) # a list of token ids	
            X2[i,] = self.dbx2.get_data(ID) # a 			
            # out1: a binary vector of total_error_types (36) elements, 
            y1[i] = self.dby1.get_data(ID)
            # out2 : a ector of one-hot-encoded vectors
            y2[i] = self.dby2.get_data(ID) #a vector of token ids (MAX_LEN:e.g., 802)

        return [X1, X2], [y1, y2]
