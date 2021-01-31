class A():
    def __init__(self):
        self.a=0
    
    def ok(self):
        a=[1002,23]
        return a
    def acc(self):
        a=self.ok()
        print('dede')
        print(a)


abc=A()
abc.acc()

