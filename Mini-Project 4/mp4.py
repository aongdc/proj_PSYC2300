import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('datasets/dataset_FR_33.xls', sheet_name=None)
data, randlists, crps35 = df['data'], df['RandomLists'], df['CRPs']

res = dict()
for row, rdata in data.iterrows():
    pr = rdata['Presentation Rate']
    res.setdefault(pr, [])
    rc = [rdata[f'recall {i}'] for i in range(1, 17)]
    res[pr].append(rc)

# pr1 = np.array(res[1000])
# pr2 = np.array(res[2000])
pr1 = pd.DataFrame(res[1000])
pr2 = pd.DataFrame(res[2000])

# pr1_cnts = dict(zip(*np.unique(pr1, return_counts=True)))
# pr2_cnts = dict(zip(*np.unique(pr1, return_counts=True)))
pr1_cnts = pr1.apply(pd.Series.value_counts).fillna(0)
pr2_cnts = pr2.apply(pd.Series.value_counts).fillna(0)

# Q1
pr1_recall = pr1_cnts.sum(axis=1) / pr1.shape[0]
pr2_recall = pr2_cnts.sum(axis=1) / pr2.shape[0]
plt.plot(range(1, 13), pr1_recall[3:], '-o', label='1000', ms=5)
plt.plot(range(1, 13), pr2_recall[3:], '-o', label='2000', ms=5)
plt.xticks(np.arange(1, 13, 1))
plt.yticks(np.arange(0, 1.1, 0.1, ))
plt.xlabel('Serial Position')
plt.ylabel('Recall Probability')
plt.title('Serial Position Curve')
plt.legend(title='Presentation Rate (ms)', title_fontsize='small', fontsize='small')
plt.show()

# Q2
pr1_pfr = pr1_cnts[0] / pr1.shape[0]
pr2_pfr = pr2_cnts[0] / pr2.shape[0]
plt.plot(range(1, 13), pr1_pfr[3:], '-o', label='1000', ms=5)
plt.plot(range(1, 13), pr2_pfr[3:], '-o', label='2000', ms=5)
plt.xticks(np.arange(1, 13, 1))
plt.yticks(np.arange(0, 0.6, 0.1, ))
plt.xlabel('Serial Position')
plt.ylabel('Recall Probability')
plt.title('Serial Position Curve')
plt.legend(title='Presentation Rate (ms)', title_fontsize='small', fontsize='small')
plt.show()

# Q3
randlists_arr = randlists.drop('list_len', axis=1).to_numpy()

# for method 3
lags_act = []
max_pos = 12
lags_poss = dict()
crps = []
for r in randlists_arr:
    # method 1
    # mask = np.ones_like(r, dtype=bool)
    # mask[idxs] = False
    # r[mask] = 0 
    
    # method 2, can replace the index itself easily
    # _, idxs = np.unique(r, return_index=True)
    # repeat_idxs = np.setxor1d(np.indices(r.shape), idxs)
    # r[repeat_idxs] = 0
    
    # method 3 by logic
    lags_r = []
    poses_r = set()
    lags_poss_r = dict()
    for idx, pos in enumerate(r):
        if idx < len(r) - 2:
            if pos not in poses_r and pos > 0 and r[idx+1] > 0:
                lags_r.append(r[idx+1] - pos)
                excl_pos = set([0, pos]).union(poses_r)
                max_k = max_pos - pos
                min_k = 1 - pos
                for k in range(min_k, max_k + 1):
                    p = k + pos
                    if p not in excl_pos:
                        lags_poss_r.setdefault(k, 0)
                        lags_poss_r[k] += 1
            else:
                lags_r.append(0)
        poses_r.add(pos)
    
    lags_r_cnt = pd.DataFrame(lags_r).apply(pd.Series.value_counts).fillna(0)
    lags_poss_cnt = pd.DataFrame.from_dict(lags_poss_r,'index')
    crp_r = lags_r_cnt / lags_poss_cnt
    crps.append(crp_r.dropna())
    
    lags_act.append(lags_r)
    for h, v in lags_poss_r.items(): 
            lags_poss.setdefault(h, 0)
            lags_poss[h] += v
            
# for method 1 and 2
# lags_act = np.diff(randlists_arr, axis=1)
# lags_act = np.where(randlists_arr[:, 1:]>0, lags_act, 0)
# lags_act = np.where(randlists_arr[:, :-1]>0, lags_act, 0)

# for all methods
lags_act_cnt = pd.DataFrame(lags_act).apply(pd.Series.value_counts).fillna(0).sum(axis=1).convert_dtypes().to_frame()
lags_poss_cnt = pd.DataFrame.from_dict(lags_poss,'index')
crps_total = (lags_act_cnt / lags_poss_cnt).dropna()

# Q4
pr1_arr = pr1.to_numpy()
lags_act = []
max_pos = 12
lags_poss = dict()
crps = []
for r in pr1_arr:
    lags_r = []
    poses_r = set()
    lags_poss_r = dict()
    for idx, pos in enumerate(r):
        if idx < len(r) - 2:
            if pos not in poses_r and pos > 0 and r[idx+1] > 0:
                lags_r.append(r[idx+1] - pos)
                excl_pos = set([0, pos]).union(poses_r)
                max_k = max_pos - pos
                min_k = 1 - pos
                for k in range(min_k, max_k + 1):
                    p = k + pos
                    if p not in excl_pos:
                        lags_poss_r.setdefault(k, 0)
                        lags_poss_r[k] += 1
            else:
                lags_r.append(0)
        poses_r.add(pos)

    lags_act.append(lags_r)
    for h, v in lags_poss_r.items(): 
            lags_poss.setdefault(h, 0)
            lags_poss[h] += v

lags_act_cnt = pd.DataFrame(lags_act).apply(pd.Series.value_counts).fillna(0).sum(axis=1).convert_dtypes().to_frame()
lags_poss_cnt = pd.DataFrame.from_dict(lags_poss,'index')
crps_total = (lags_act_cnt / lags_poss_cnt).dropna()

pr2_arr = pr2.to_numpy()
lags_act = []
max_pos = 12
lags_poss = dict()
crps = []
for r in pr2_arr:
    lags_r = []
    poses_r = set()
    lags_poss_r = dict()
    for idx, pos in enumerate(r):
        if idx < len(r) - 2:
            if pos not in poses_r and pos > 0 and r[idx+1] > 0:
                lags_r.append(r[idx+1] - pos)
                excl_pos = set([0, pos]).union(poses_r)
                max_k = max_pos - pos
                min_k = 1 - pos
                for k in range(min_k, max_k + 1):
                    p = k + pos
                    if p not in excl_pos:
                        lags_poss_r.setdefault(k, 0)
                        lags_poss_r[k] += 1
            else:
                lags_r.append(0)
        poses_r.add(pos)

    lags_act.append(lags_r)
    for h, v in lags_poss_r.items(): 
            lags_poss.setdefault(h, 0)
            lags_poss[h] += v

lags_act_cnt = pd.DataFrame(lags_act).apply(pd.Series.value_counts).fillna(0).sum(axis=1).convert_dtypes().to_frame()
lags_poss_cnt = pd.DataFrame.from_dict(lags_poss,'index')
crps_total = (lags_act_cnt / lags_poss_cnt).dropna()