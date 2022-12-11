"""We define a fixed size queue object."""


class FixedQueue:
    def __init__(self, maxsize = 5):
        self.maxsize = maxsize
        self.elements = []
    
    def put(self, other):
        if len(self.elements) >= self.maxsize:
            # push the one off the end, then insert at the start.
            self.elements.pop(-1)
            self.elements.insert(0, other)
        else:
            self.elements.insert(0, other)
    
    def __getitem__(self, i):
        return self.elements[i]
