import torch
import numpy as np
import pickle

torch.set_default_dtype(torch.double)

def prep_pickle(data, dst):
    with open(f'{data}', 'rb') as f:
        x = pickle.load(f)
    try:
        x_tensor = torch.from_numpy(x)
    except TypeError:
        x_tensor = np.array(x)                  # convert to numpy in the case of simple lists 
        print(x_tensor.shape)

    torch.save(x_tensor, dst)

def prep_numpy(data, dst):
    x = np.load(data)
    x_tensor = torch.from_numpy(x)

    torch.save(x_tensor, dst)


if __name__ == '__main__':
    prep_pickle("../test_data/original_pickle_data.pickle", "../test_data/pickle_data_after_prep.p")
    print('Test 1 pass')

    prep_pickle("../test_data/array_30_200.pkl", "../test_data/array_30_200_after_prep.p")
    print('Test 2 pass')

    prep_numpy('../test_data/random_numpy_data.npy', '../test_data/numpy_data_after_prep.p')
    print('Test 3 pass')