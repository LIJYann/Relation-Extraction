#encoding=utf-8

from gensim.models import word2vec
import numpy as np


class Embedding(object):
    def __init__(self, modelPath,position_size):
        self.model = word2vec.Word2Vec.load(modelPath)
        self.position_size = position_size

    def takeSecond(self, elem):
        return elem[1]

    def position_Embedding(self, inputs):
        seq_len = np.shape(inputs)[1]
        position_j = 1. / np.power(10000., 2 * np.arange(self.position_size / 2, dtype=np.float32) / self.position_size)
        position_j = np.expand_dims(position_j, 0)
        position_i0 = inputs.astype(np.float32)
        position_i = np.reshape(position_i0, (seq_len, -1))
        position_ij = np.matmul(position_i, position_j)
        position_ij = np.concatenate([np.cos(position_ij), np.sin(position_ij)], 1)
        position_embedding = np.expand_dims(position_ij, 0) + np.zeros((seq_len, self.position_size))
        return position_embedding

    def transform(self, path, res, res_type):
        
        f = open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        num_relation = []
        L = []
        row = -1
        for line in lines:
            if line.split('\t')[0] != '':
                row = row + 1
                if len(line.split('\t')) > 1 and line.split('\t')[1] == '代词':
                    L.append(line.split('\t')[2])
                else:
                    L.append(line.split('\t')[0])
                if line.split('\t')[0] != '。':
                    for col in range(3, len(line.split('\t'))):
                        if len(line.split('\t')) > 3 and line.split('\t')[col] != '' and line.split('\t')[col] != '\n':
                            num_relation.append([row, col, line.split('\t')[col]])
                else:
                    # print(num_relation)
                    if len(num_relation) > 0:
                        num_relation.sort(key=self.takeSecond)
                        for i in range(len(num_relation) // 2):
                            T1 = []
                            T2 = []
                            count_first = 0
                            # print(len(num_relation)//2)
                            for i_n in range(2):
                                # print([rows,cols,content])
                                [rows, cols, content] = num_relation[2 * i + i_n]
                                for j in range(row + 1):
                                    if 's' in content:
                                        T1.append(j - rows)
                                        type_class = int(content.replace('s', ''))
                                        row1 = rows
                                    else:
                                        T2.append(j - rows)
                                        row2 = rows
                            temps = []
                            if row1 < row2:
                                # type_class = int(type_class)
                                pass
                            else:
                                type_class = int(type_class) + 5
                                temps = T1
                                T1 = T2
                                T2 = T1

                            t1 = np.reshape(np.asarray(T1, dtype=np.int32), (1, -1))
                            t2 = np.reshape(np.asarray(T2, dtype=np.int32), (1, -1))
                            # print(t2)
                            pos_embed1 = self.position_Embedding(t1, self.position_size)
                            pos_embed2 = self.position_Embedding(t2, self.position_size)
                            res_matrix = np.zeros((200,320))
                            for t, words in enumerate(L):
                                # print(str(pos_embed1[0][i].eval(session=sess)))
                                array1 = pos_embed1[0][t]
                                array2 = pos_embed2[0][t]
                                res_matrix[t][0:self.position_size-1] = array1
                                res_matrix[t][self.position_size:self.position_size*2 - 1] = array2
                                if words not in self.model:
                                    words = 'UNK'
                                else:
                                    pass
                                res_matrix[t][self.position_size*2:319] = self.model.wv[words]
                            # print('it is done\n')
                            res_type.append(type_class-1)
                            res.append(res_matrix)
        return res, res_type

    def transform_nonnoise(self, path, res, res_type):

        f = open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        num_relation = []
        L = []
        row = -1
        for line in lines:
            if line.split('\t')[0] != '':
                row = row + 1
                if len(line.split('\t')) > 1 and line.split('\t')[1] == '代词':
                    L.append(line.split('\t')[2])
                else:
                    L.append(line.split('\t')[0])
                if line.split('\t')[0] != '。':
                    for col in range(3, len(line.split('\t'))):
                        if len(line.split('\t')) > 3 and line.split('\t')[col] != '' and line.split('\t')[col] != '\n':
                            num_relation.append([row, col, line.split('\t')[col]])
                else:
                    # print(num_relation)
                    if len(num_relation) > 0:
                        num_relation.sort(key=self.takeSecond)
                        for i in range(len(num_relation) // 2):
                            T1 = []
                            T2 = []
                            count_first = 0
                            # print(len(num_relation)//2)
                            [rows1, cols1, content1] = num_relation[2 * i + 0]
                            [rows2, cols2, content2] = num_relation[2 * i + 1]
                            for i_n in range(2):
                                # print([rows,cols,content])
                                [rows, cols, content] = num_relation[2 * i + i_n]
                                for j in range(row + 1):
                                    if 's' in content:
                                        T1.append(j - rows)
                                        type_class = int(content.replace('s', ''))
                                        row1 = rows
                                    else:
                                        T2.append(j - rows)
                                        row2 = rows
                            temps = []
                            if row1 < row2:
                                # type_class = int(type_class)
                                pass
                            else:
                                type_class = int(type_class) + 5
                                temps = T1
                                T1 = T2
                                T2 = T1

                            t1 = np.reshape(np.asarray(T1, dtype=np.int32), (1, -1))
                            t2 = np.reshape(np.asarray(T2, dtype=np.int32), (1, -1))
                            # print(t2)
                            pos_embed1 = self.position_Embedding(t1, self.position_size)
                            pos_embed2 = self.position_Embedding(t2, self.position_size)
                            res_matrix = np.zeros((200, 320))
                            for t, words in enumerate(L):
                                # print(str(pos_embed1[0][i].eval(session=sess)))
                                array1 = pos_embed1[0][t]
                                array2 = pos_embed2[0][t]
                                res_matrix[t][0:self.position_size - 1] = array1
                                res_matrix[t][self.position_size:self.position_size * 2 - 1] = array2
                                if words not in self.model:
                                    words = 'UNK'
                                else:
                                    pass
                                res_matrix[t][self.position_size * 2:319] = self.model.wv[words]
                            # print('it is done\n')
                            if row1 != 0:
                                res_matrix[0:row1 - 1] = np.zeros((row1, 320))
                            if row2 != 199:
                                res_matrix[row2 + 1:199] = np.zeros((199 - row2, 320))
                            res_type.append(type_class - 1)
                            res.append(res_matrix)
        return res, res_type

    def transform_nonnoise(self, path, res, res_type):

        f = open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        num_relation = []
        L = []
        row = -1
        for line in lines:
            if line.split('\t')[0] != '':
                row = row + 1
                if len(line.split('\t')) > 1 and line.split('\t')[1] == '代词':
                    L.append(line.split('\t')[2])
                else:
                    L.append(line.split('\t')[0])
                if line.split('\t')[0] != '。':
                    for col in range(3, len(line.split('\t'))):
                        if len(line.split('\t')) > 3 and line.split('\t')[col] != '' and line.split('\t')[col] != '\n':
                            num_relation.append([row, col, line.split('\t')[col]])
                else:
                    # print(num_relation)
                    if len(num_relation) > 0:
                        num_relation.sort(key=self.takeSecond)
                        for i in range(len(num_relation) // 2):
                            T1 = []
                            T2 = []
                            count_first = 0
                            # print(len(num_relation)//2)
                            [rows1, cols1, content1] = num_relation[2 * i + 0]
                            [rows2, cols2, content2] = num_relation[2 * i + 1]
                            for i_n in range(2):
                                # print([rows,cols,content])
                                [rows, cols, content] = num_relation[2 * i + i_n]
                                for j in range(row + 1):
                                    if 's' in content:
                                        T1.append(j - rows)
                                        type_class = int(content.replace('s', ''))
                                        row1 = rows
                                    else:
                                        T2.append(j - rows)
                                        row2 = rows
                            temps = []
                            if row1 < row2:
                                # type_class = int(type_class)
                                pass
                            else:
                                type_class = int(type_class) + 5
                                temps = T1
                                T1 = T2
                                T2 = T1

                            t1 = np.reshape(np.asarray(T1, dtype=np.int32), (1, -1))
                            t2 = np.reshape(np.asarray(T2, dtype=np.int32), (1, -1))
                            # print(t2)
                            pos_embed1 = self.position_Embedding(t1, self.position_size)
                            pos_embed2 = self.position_Embedding(t2, self.position_size)
                            res_matrix = np.zeros((200, 320))
                            for t, words in enumerate(L):
                                # print(str(pos_embed1[0][i].eval(session=sess)))
                                array1 = pos_embed1[0][t]
                                array2 = pos_embed2[0][t]
                                res_matrix[t][0:self.position_size - 1] = array1
                                res_matrix[t][self.position_size:self.position_size * 2 - 1] = array2
                                if words not in self.model:
                                    words = 'UNK'
                                else:
                                    pass
                                res_matrix[t][self.position_size * 2:319] = self.model.wv[words]
                            # print('it is done\n')
                            if row1 != 0:
                                res_matrix[0:row1 - 1] = np.zeros((row1, 320))
                            if row2 != 199:
                                res_matrix[row2 + 1:199] = np.zeros((199 - row2, 320))
                            res_type.append(type_class - 1)
                            res.append(res_matrix)
        return res, res_type

    def load_data(self, res, res_type, division=5):
        data_train=[]
        label_train=[]
        data_test=[]
        label_test=[]
        for i, matrix in enumerate(res):
            if i%division == 1:
                label_test.append(res_type[i])
                data_test.append(matrix)
            else:
                label_train.append(res_type[i])
                data_train.append(matrix)
        data_array_test = np.array(data_test)
        label_array_test = np.array(label_test)
        state = np.random.get_state()
        np.random.shuffle(data_array_test)
        np.random.set_state(state)
        np.random.shuffle(label_array_test)

        data_array_train = np.array(data_train)
        label_array_train = np.array(label_train)
        state = np.random.get_state()
        np.random.shuffle(data_array_train)
        np.random.set_state(state)
        np.random.shuffle(label_array_train)
        return data_array_train, label_array_train, data_array_test, label_array_test

