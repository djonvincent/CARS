import numpy as np

user_ids = []
item_ids = []
context_ids = []

def id_to_int(i, ls):
    try:
        return ls.index(i)
    except ValueError:
        ls.append(i)
        return len(ls)-1

ds_raw = np.loadtxt(
    'frappe.csv',
    np.float64,
    delimiter=',',
    converters={
        0: lambda c: id_to_int(c, user_ids),
        1: lambda c: id_to_int(c, item_ids),
        3: lambda c: -1 if c == b"unknown" else id_to_int(c, context_ids)
    },
    skiprows=1,
    usecols=(0,1,3,2)
)

ds_raw = ds_raw[ds_raw[:,2] != -1]

while True:
    select = np.bincount(ds_raw[:,0].astype(np.int64))[ds_raw[:,0].astype(np.int64)] >= 100
    ds_raw = ds_raw[select]
    select2 = np.bincount(ds_raw[:,1].astype(np.int64))[ds_raw[:,1].astype(np.int64)] >= 100
    ds_raw = ds_raw[select2]
    select3 = ds_raw[:,3] <= 50
    ds_raw = ds_raw[select3]
    if all(select) and all(select2) and all(select3):
        break
'''
user_max = np.zeros(int(np.amax(ds_raw[:,0]))+1)
for i in np.unique(ds_raw[:,0].astype(np.int64)):
    user_max[i] = np.amax(ds_raw[:,3][ds_raw[:,0]==i])
ds_raw[:,3] = np.multiply(ds_raw[:,3], 5/user_max[ds_raw[:,0].astype(np.int32)])
'''
percentiles = np.percentile(ds_raw[:,3], [20,40,60,80])
for r in ds_raw:
    r[3] = np.sum(percentiles < r[3]) + 1
print(len(ds_raw))
print(len(np.unique(ds_raw[:,0])),len(np.unique(ds_raw[:,1])),len(np.unique(ds_raw[:,2])))
print(len(ds_raw)/(len(np.unique(ds_raw[:,0]))*len(np.unique(ds_raw[:,1]))*len(np.unique(ds_raw[:,2]))))

