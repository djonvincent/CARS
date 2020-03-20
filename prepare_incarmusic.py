import numpy as np

user_ids = []
hotel_ids = []
context_ids = []

def id_to_int(i, ls):
    try:
        return ls.index(i)
    except ValueError:
        ls.append(i)
        return len(ls)-1

ds_raw = np.loadtxt(
    'Data_InCarMusic.csv',
    np.int64,
    delimiter=',',
    converters={
        0: lambda c: id_to_int(c, user_ids),
        1: lambda c: id_to_int(c, hotel_ids),
        4: lambda c: -1 if c == b"NA" else id_to_int(c, context_ids)
    },
    skiprows=1,
    usecols=(0,1,4,2)
)

ds_raw = ds_raw[ds_raw[:,2] != -1]
print(f'Contexts: {len(np.unique(ds_raw[:,2]))}')
print(len(ds_raw)/(len(np.unique(ds_raw[:,0]))*len(np.unique(ds_raw[:,1]))*len(np.unique(ds_raw[:,2]))))

