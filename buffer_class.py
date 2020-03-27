class Buffer:
    def __init__(self, size):
        self.size = size
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        string = '['
        for i in self.data:
            string += str(i) + ','
        string = string[:-1]
        string += ']'
        return string

    def append(self, item):
        length = len(self.data)
        if length < self.size:
            self.data.append(item)
        else:
            self.data.pop(0)
            self.data.append(item)

    def as_list(self):
        return self.data