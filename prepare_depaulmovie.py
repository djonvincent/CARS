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
    'depaulmovie.csv',
    np.int64,
    delimiter=',',
    converters={
        0: lambda c: id_to_int(c, user_ids),
        1: lambda c: id_to_int(c, hotel_ids),
        5: lambda c: -1 if c == b"NA" else id_to_int(c, context_ids)
    },
    skiprows=1,
    usecols=(0,1,5,2)
)

ds_raw = ds_raw[ds_raw[:,2] != -1]

