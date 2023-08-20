import pickle
import pandas as pd

def read_pickle(f):
    return pickle.load(open(f, 'rb'))

def write_pickle(obj, f):
    pickle.dump(obj, open(f, 'wb'))

def load_network(network):
    return pd.read_pickle(network)