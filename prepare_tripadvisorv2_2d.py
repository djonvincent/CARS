import numpy as np
from tensorly.decomposition import parafac, robust_pca, matrix_product_state
from tensorly import kruskal_to_tensor
from scipy.sparse.linalg import svds

user_ids = []
hotel_ids = []

def id_to_int(i, ls):
    try:
        return ls.index(i)
    except ValueError:
        ls.append(i)
        return len(ls)-1

ds_raw = np.loadtxt(
    'tripadvisorv2.csv',
    np.int64,
    delimiter=',',
    converters={
        0: lambda c: id_to_int(c, user_ids),
        1: lambda c: id_to_int(c, hotel_ids),
    },
    skiprows=1,
    usecols=(0,1,2)
)

ds = ds_raw[:,:]
while True:
    hotel_count = dict(zip(*np.unique(ds[:,1], return_counts=True)))
    select = [hotel_count[x[1]] >= 5 for x in ds]
    print(np.sum(select))
    ds = ds[select]
    user_count = dict(zip(*np.unique(ds[:,0], return_counts=True)))
    select2 = [user_count[x[0]] >= 5 for x in ds]
    print(np.sum(select2))
    ds = ds[select2]
    #type_count = dict(zip(*np.unique(ds[:,3], return_counts=True)))
    #select3 = [user_count[x[3]] >= 5 for x in ds]
    #print(np.sum(select3))
    #ds = ds[select3]
    if all(select2) and all(select):
        break

tensor = np.zeros((len(np.unique(ds[:,0])), len(np.unique(ds[:,1]))))
print(np.shape(tensor))
for r in ds:
    i = np.argwhere(np.unique(ds[:,0])==r[0])[0][0]
    j = np.argwhere(np.unique(ds[:,1])==r[1])[0][0]
    tensor[i][j] = r[2]

U, sigma, Vt = svds(tensor, k=300)
sigma = np.diag(sigma)
pred = np.dot(np.dot(U, sigma), Vt)
print(np.mean(np.abs(tensor[tensor>0] - pred[tensor>0])))
