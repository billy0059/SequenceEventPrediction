import random
import numpy as np
class DataHandler : 
    def __init__(self, input_data, training_percentage = 0.8, testing_percentage = 0.2):
        temp = []
        
        for i in range(int (testing_percentage*len(input_data))):
            pattern = random.choice(input_data)
            temp.append(pattern)
            input_data.remove(pattern)
            
        self.input = input_data
        self.index = self.getIndex(input_data)
        #input_data = 
        self.training_data = input_data
        self.testing_data = temp
        
        self.training_cursor = 0
        self.testing_cursor = 0
        self.training_step_cursor = [random.randint(0,(len(input_data[i])))  for i in range(len(input_data))]
        self.testing_step_cursor = [random.randint(0,(len(temp[i]))) for i in range(len(temp))]
        
    def howManyData(self, time_step):
        sum = 0
        for pattern in self.training_data:
            sum += len(pattern) - time_step + 1
        return sum
        
    def Patternfrequent(self):
        temp = {i:0 for i in self.index}
        for pattern in self.training_data:
            for event in pattern:
                temp[event] += 1
        for pattern in self.testing_data:
            for event in pattern:
                temp[event] += 1
        return temp
    def getIndex(self, input): 
        index = {}
        i = 1
        for pattern in self.input:
            for action in pattern:
                if index.get(action) == None:
                    index[action] = i
                    i+=1
        index["(PURCHASE!!!)"] = i
        index["Ignore"] = 0
        return index 
    
    def pattern_input_shape (self, index_value): 
        data = np.zeros(shape = (len(self.index)))
        data[index_value] = 1
        return data    
    
    def getKeyByValue(self, dic, value): 
        for key, _value in dic.items():
            if value == _value:
                return key
    
    def getTraining(self, batch_size, time_step):
        batch = []
        true = []
        for i in range(batch_size):
            pattern = []
            if self.training_step_cursor[self.training_cursor] + time_step > len(self.training_data[self.training_cursor]):
                self.training_step_cursor[self.training_cursor] = 0
            pattern.append(self.eventCountVector(self.training_data[self.training_cursor],self.training_step_cursor[self.training_cursor]))
            for j in range(time_step):
                pattern.append(self.pattern_input_shape(self.index[self.training_data[self.training_cursor][self.training_step_cursor[self.training_cursor]+j]]))
            batch.append(pattern)
            try:
                true.append(self.pattern_input_shape(self.index[self.training_data[self.training_cursor][self.training_step_cursor[self.training_cursor]+time_step]]))
            except:
                true.append(self.pattern_input_shape(self.index['(PURCHASE!!!)']))
            self.training_step_cursor[self.training_cursor] += 1
            if self.training_cursor == len(self.training_data) -1:
                self.training_cursor = 0
            else:
                self.training_cursor += 1
        return batch, true
    def getTesting(self, batch_size, time_step):
        batch = []
        true = []
        for i in range(batch_size):
            pattern = []
            if self.testing_step_cursor[self.testing_cursor] + time_step > len(self.testing_data[self.testing_cursor]):
                self.testing_step_cursor[self.testing_cursor] = 0
            pattern.append(self.eventCountVector(self.testing_data[self.testing_cursor],self.testing_step_cursor[self.testing_cursor]))
            for j in range(time_step):
                pattern.append(self.pattern_input_shape(self.index[self.testing_data[self.testing_cursor][self.testing_step_cursor[self.testing_cursor]+j]]))
            batch.append(pattern)
            try:
                true.append(self.pattern_input_shape(self.index[self.testing_data[self.testing_cursor][self.testing_step_cursor[self.testing_cursor]+time_step]]))
            except:
                true.append(self.pattern_input_shape(self.index['(PURCHASE!!!)']))
            self.testing_step_cursor[self.testing_cursor] += 1
            if self.testing_cursor == len(self.testing_data) -1:
                self.testing_cursor = 0
            else:
                self.testing_cursor += 1
        return batch, true
        
    def getTrainingPerUUID(self, batch_size, time_step):
        batch = []
        true = []
        for i in range(batch_size):
            pattern = []
            if self.training_step_cursor[self.training_cursor] + time_step > len(self.training_data[self.training_cursor]):
                self.training_step_cursor[self.training_cursor] = 0
            for j in range(time_step):
                pattern.append(self.pattern_input_shape(self.index[self.training_data[self.training_cursor][self.training_step_cursor[self.training_cursor]+j]]))
            batch.append(pattern)
            try:
                true.append(self.pattern_input_shape(self.index[self.training_data[self.training_cursor][self.training_step_cursor[self.training_cursor]+time_step]]))
            except:
                true.append(self.pattern_input_shape(self.index['(PURCHASE!!!)']))
            self.training_step_cursor[self.training_cursor] += 1
        if self.training_cursor == len(self.training_data) -1:
            self.training_cursor = 0
        else:
            self.training_cursor += 1
        return batch, true
    def getTestingPerUUID(self, batch_size, time_step):
        batch = []
        true = []
        for i in range(batch_size):
            pattern = []
            if self.testing_step_cursor[self.testing_cursor] + time_step > len(self.testing_data[self.testing_cursor]):
                self.testing_step_cursor[self.testing_cursor] = 0
            for j in range(time_step):
                pattern.append(self.pattern_input_shape(self.index[self.testing_data[self.testing_cursor][self.testing_step_cursor[self.testing_cursor]+j]]))
            batch.append(pattern)
            try:
                true.append(self.pattern_input_shape(self.index[self.testing_data[self.testing_cursor][self.testing_step_cursor[self.testing_cursor]+time_step]]))
            except:
                true.append(self.pattern_input_shape(self.index['(PURCHASE!!!)']))
            self.testing_step_cursor[self.testing_cursor] += 1
        if self.testing_cursor == len(self.testing_data) -1:
            self.testing_cursor = 0
        else:
            self.testing_cursor += 1
        return batch, true  
    
    def eventCountVector(self, pattern, cursor):
        count_vector = [0 for i in range(len(self.index))]
        for i in range(cursor):
            count_vector[self.index[pattern[i]]] += 1
        return count_vector
            
    