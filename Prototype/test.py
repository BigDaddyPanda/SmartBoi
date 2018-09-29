import pickle
class Foo:
    attr = 'A class attribute'

print(pickle.dumps(Foo))
