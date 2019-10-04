import random

class ReplayBuffer():
    '''
    The transitions are stored in a list (self.experiences) and a pointer indicates the position where
    the next transition will be stored. This pointer resets when the maximum capacity is reached, so that
    new transitions erase old ones without discrimination. Transitions are sampled uniformly from the list.
    '''
    def __init__(self, 
                capacity = 50000):
        self.capacity = capacity
        self.experiences = []        
        self.pointer = 0
    
    def store(self, experience):
        if len(self.experiences) < self.capacity:
            self.experiences.append(None)
        self.experiences[self.pointer] = experience
        self.pointer = (self.pointer + 1) % self.capacity
    
    def sample_transitions(self, batch_size):
        '''
        Returns a list whose elements are lists of: observations, actions, rewards, next_observations, 
        and termination signals. So, e.g., transposed_batch[2] is a list of only rewards.
        '''
        if batch_size < len(self.experiences):
            batch =  random.sample(self.experiences, int(batch_size)) 
        else:
            batch = random.sample(self.experiences, len(self.experiences))
        transposed_batch = list(map(list, zip(*batch)))
        return transposed_batch

    def retrieve(self):
        return self.experiences.copy()
