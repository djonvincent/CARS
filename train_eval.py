import numpy as np
from tensorly.decomposition import parafac, tucker
from tensorly import kruskal_to_tensor
import tensortools as tt

def train_eval(ds_raw, train_test_ratio, rank, num_tests):
    total_mae = 0
    total_precision = 0
    total_recall = 0
    total_f_measure = 0

    users = np.unique(ds_raw[:,0])
    items = np.unique(ds_raw[:,1])
    contexts = np.unique(ds_raw[:,2])
    print(len(ds_raw))
    print(len(users), len(items), len(contexts))

    #Average ratings for identical user-item-context
    ds = []
    totals = np.zeros((len(users), len(items), len(contexts)))
    counts = np.zeros((len(users), len(items), len(contexts)))
    for r in ds_raw:
        i = np.argwhere(users==r[0])[0][0]
        j = np.argwhere(items==r[1])[0][0]
        k = np.argwhere(contexts==r[2])[0][0]
        totals[i][j][k] += r[3]
        counts[i][j][k] += 1

    for (i,j,k) in np.argwhere(counts > 0):
        ds.append([i,j,k,totals[i][j][k]/counts[i][j][k]])
    ds = np.asarray(ds)
    density = np.sum(counts > 0) / (len(users) * len(items) * len(contexts))
    print(f'Density of train tensor: {density}')
    #return ds, ds_raw

    train_n = int(len(ds)*train_test_ratio)
    print(f'Train samples: {train_n}')
    print(f'Test samples: {len(ds)-train_n}')
    for test in range(num_tests):
        print(f'Running test {test+1}')
        np.random.shuffle(ds)
        train = ds[:train_n]
        test = ds[train_n:]

        tensor_train = np.zeros((len(users), len(items), len(contexts)))
        '''
        while True:
            hotel_count = dict(zip(*np.unique(ds[:,1], return_counts=True)))
            select = [hotel_count[x[1]] >= 41 for x in ds]
            print(np.sum(select))
            ds = ds[select]
            user_count = dict(zip(*np.unique(ds[:,0], return_counts=True)))
            select2 = [user_count[x[0]] >= 41 for x in ds]
            print(np.sum(select2))
            ds = ds[select2]
            #type_count = dict(zip(*np.unique(ds[:,3], return_counts=True)))
            #select3 = [user_count[x[3]] >= 5 for x in ds]
            #print(np.sum(select3))
            #ds = ds[select3]
            if all(select2) and all(select):
                break
        '''
        #print(np.shape(tensor))
        #print(len(ds)/tensor.size)
        #ratings_mean = np.mean(ds[:,2])
        for r in train:
            i = int(r[0])
            j = int(r[1])
            k = int(r[2])
            tensor_train[i][j][k] = r[3]

        factors = tt.mcp_als(tensor_train, rank=11, mask=tensor_train>0, verbose=False)
        pred = kruskal_to_tensor((None, factors.factors.factors))
        #factors = parafac(tensor_train, rank=rank)
        #pred = kruskal_to_tensor(factors)
        mae = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for r in test:
            i = int(r[0])
            j = int(r[1])
            k = int(r[2])
            y = r[3]
            p = pred[i][j][k]
            mae += abs(y - p)
            tp += y >= 2.5 and p >= 2.5
            fp += y < 2.5 and p >= 2.5
            tn += y < 2.5 and p < 2.5
            fn += y >= 2.5 and p < 2.5

        mae /= len(test)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f = 2*precision*recall/(precision+recall)
        print(f'\tMAE: {mae}')
        print(f'\tPrecision: {precision}')
        print(f'\tRecall: {recall}')
        print(f'\tF-measure: {f}')
        total_mae += mae
        total_precision += precision
        total_recall += recall
        total_f_measure += f
    print(f'MAE: {total_mae/num_tests}')
    print(f'Precision: {total_precision/num_tests}')
    print(f'Recall: {total_recall/num_tests}')
    print(f'F-measure: {total_f_measure/num_tests}')

if __name__ == "__main__":
    from prepare_depaulmovie import ds_raw as ds_movie
    from prepare_frappe import ds_raw as ds_frappe
    print("DePaulMovie")
    #train_eval(ds_movie, 0.99, 30, 5)
    print("Frappe")
    train_eval(ds_frappe, 0.95, 50, 5)
