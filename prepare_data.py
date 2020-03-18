import numpy as np
from tensorly.decomposition import parafac

user_ids = []
hotel_ids = []

def id_to_int(i, ls):
    try:
        return ls.index(i)
    except ValueError:
        ls.append(i)
        return len(ls)-1

ds = np.loadtxt(
    'tripadvisorv2.csv',
    np.uint64,
    delimiter=',',
    converters={
        0: lambda c: id_to_int(c, user_ids),
        1: lambda c: id_to_int(c, hotel_ids),
        8: lambda c: [
            b'FAMILY',b'COUPLES',b'BUSINESS',b'SOLO',b'FRIENDS'
        ].index(c)
    },
    skiprows=1,
    usecols=(0,1,2,8)
)

tensor = np.zeros((len(user_ids), len(hotel_ids), 5))
for r in ds:
    tensor[r[0]][r[1]][r[3]] = r[2]

factors = parafac(tensor, rank=3)
print(factors)
