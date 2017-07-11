import os
import numpy as np
from keras.preprocessing import sequence

class Protein_Dataset:
    """
    API for Protein dataset. Plus training utilities.
    """
    def __init__(self, data_path, maxlen=400, stride=100):
        """
        train_path: path to training folder
        test_path: path to testing folder
        maxlen: length to which all sequnces will be padded to
        stride: number of seq elements that will be taken into last batch from previous one
        """
        self.train = self._load_data(os.path.join(data_path, "train"))
        self.test = self._load_data(os.path.join(data_path, "test"))
        
        if os.path.isfile(os.path.join(data_path, 'acid.npy')):
            self.acid_table = list(np.load(os.path.join(data_path, 'acid.npy')))
        else:
            self.acid_table = list(np.unique(list("".join(self.train[:,0]))))
            np.save(os.path.join(data_path, 'acid'), acid_table)
            
        if os.path.isfile(os.path.join(data_path, 'class.npy')):
            self.class_table = list(np.load(os.path.join(data_path, 'class.npy')))
        else:
            self.class_table = list(np.unique(list("".join(self.train[:,1]))))
            np.save(os.path.join(data_path, 'class'), class_table)
        
        self.train = self._translate_data(self.train)
        self.test = self._translate_data(self.test)
        
        self.maxlen = maxlen
        self.stride = stride
        
        
    def _read_file(self, path):
        with open(path) as f:
            a = f.readlines()
        return a
    
    def _load_data(self, dir_path):
        data = []

        for f in set(list(map(lambda x: ".".join(x.split('.')[:-1]), os.listdir(dir_path)))):
            dssp = self._read_file(os.path.join(dir_path, f+".dssp"))[1].strip("\n")
            fasta = self._read_file(os.path.join(dir_path, f+".fasta"))[1].strip("\n")
            data.append([fasta, dssp])

        return np.array(data)
    
    def _translate_data(self, data):
    
        acid_code = lambda x: np.array([self.acid_table.index(y) for y in x])
        class_code = lambda x: np.array([self.class_table.index(y) + 1 for y in x])

        return np.array(list(map(lambda x: [acid_code(x[0]), class_code(x[1])], data)))
    
    def _tokens_to_classes(self, seq, seq_len=None):
        if seq_len == None:
            seq_len = len(seq)
        return "".join(list(map(lambda x: self.class_table[int(x)], seq)))[:seq_len]
    
    def tokens_to_string(self, pred_raw, seq_len=None):
        pred = pred_raw - 1
        pred = np.array(list(map(lambda x: 0 if x == -1 else x, pred)))
        return self._tokens_to_classes(pred, seq_len)
    
    def split_seq(self, seq):
        seq_len = seq.shape[0]

        temp = []

        if seq_len < self.maxlen:
            temp.append(seq)

        else:
            full_batches = seq_len // self.maxlen

            for i in range(full_batches):
                idx = slice(i * self.maxlen, (i+1) * self.maxlen)
                temp.append(seq[idx])

            elements_left = seq_len % self.maxlen

            if elements_left:
                start_element = elements_left + self.stride

                if start_element > self.maxlen:
                    start_element = self.maxlen

                temp.append(seq[-start_element:])
        return temp
    
    def split_pair_seq(self, pair_seq):
        
        assert pair_seq[0].shape == pair_seq[1].shape

        return list(map(list, zip(self.split_seq(pair_seq[0]), self.split_seq(pair_seq[1]))))
    
    def get_prepared_data(self, train=True, one_hot=True):
        
        if train:
            data = self.train
        else:
            data = self.test
        
        # Dataset preparation
        splited = list(map(self.split_pair_seq, data))

        prep_data = []

        for seq in splited:
            prep_data.extend(seq)

        prep_data = np.array(prep_data)
        
        x_s = sequence.pad_sequences(prep_data[:, 0], padding="post", value=-1, maxlen=self.maxlen)
        y_s = sequence.pad_sequences(prep_data[:, 1], padding="post", maxlen=self.maxlen)
        
        
        if one_hot:
            acid_len = len(self.acid_table)
            output_len = len(self.class_table) + 1
            
            x_s = np.array(list(map(lambda x: self._one_hot(x, acid_len), x_s)))
            y_s = np.array((list(map(lambda x: self._one_hot(x, output_len), y_s))))
            
            
            
        return x_s, y_s
            
    def dynamic_iter(self, batchsize, train=True, shuffle=True, one_hot=False):
        
        x,y = self.get_prepared_data(train, one_hot)
            
        # Batching
        index = list(range(len(x)))

        if shuffle:
            np.random.shuffle(index)
            
        for i in range(0, len(x) - batchsize + 1, batchsize):

            x_batch = x[index[i:i+batchsize]]
            y_batch = y[index[i:i+batchsize]]

            seq_len = []
            for seq in x_batch:
                l = np.sum(np.not_equal(seq, -1))
                seq_len.append(l)
            seq_len = np.array(seq_len)

            yield x, y, seq_len
            
    def _one_hot(self, seq, num_classes=None):
        if num_classes == None:
            num_classes = max(seq) + 1

        ret_matr = np.zeros([seq.shape[0], num_classes])

        for i, el in enumerate(seq):
            if el == -1:
                continue
            ret_matr[i, el] = 1

        return ret_matr