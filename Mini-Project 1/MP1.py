import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/dataset_miniproject1_45.csv')
thresholds = range(1, 8)

# remove timeout trials
data = data.dropna()

def hitrate(t_res):
    hr = t_res['hits'] / (t_res['hits'] + t_res['misses'])
    return hr  #, round(hr, 2)

def farate (t_res):
    far = t_res['fas'] / (t_res['fas'] + t_res['rejs'])
    return far  #, round(far, 2)

# dataset average HR, FAR
def q1a(data):
    res = {c: {'hits': 0, 'misses': 0, 'fas': 0, 'rejs': 0} for c in thresholds}
    for _, row in data.iterrows():
        # vary confidence thresholds
        varied = [row['Confidence rating'] > thres for thres in thresholds]
        
        if row['Study-test lag'] > 0:
            for i, v in enumerate(varied):
                if v:
                    res[i+1]['hits'] += 1
                else:
                    res[i+1]['misses'] += 1
        else:
            for i, v in enumerate(varied):
                if v: 
                    res[i+1]['fas'] += 1
                else:
                    res[i+1]['rejs'] += 1
    return res

# across subject average HR, FAR
def q1b(data):
    # stats for each subject for each C
    subj_data = dict()
    for idx, sdata in data.groupby('Subject'):
        subj_data[idx] = q1a(sdata)

    # HR, FAR for each subject for each C
    subj_res = dict()
    for s, dt in subj_data.items():
        subj_res[s] = {thres: {'hr': 0, 'far': 0} for thres in thresholds}
        for k, v in dt.items():
            subj_res[s][k]['hr'] = hitrate(v)
            subj_res[s][k]['far'] = farate(v)

    # aggregate HR, FAR across subjects for each C
    hrs = {thres: [] for thres in thresholds}
    fars = {thres: [] for thres in thresholds}
    for thres in thresholds:
        for s in subj_res.keys():
            hrs[thres].append(subj_res[s][thres]['hr'])
            fars[thres].append(subj_res[s][thres]['far'])
    
    # average HR, FAR across subjects for each C  
    hrsf = {hrk: sum(hrv) / len(hrv) for hrk, hrv in hrs.items()}
    farsf = {fark: sum(farv) / len(farv) for fark, farv in fars.items()}
    
    return subj_res, hrs, fars, hrsf, farsf

def q2(data):
    # stats for each subject for each C
    subj_res = q1b(data)[0]

    # stats for first 10 subjects for each C
    subj_ten = {s: v for s, v in subj_res.items() if s in list(subj_res)[:10]}
    
    # HR, FAR for first 10 subjects for each C
    x_ten = dict()
    y_ten = dict()
    for s, d in subj_ten.items():
        x_ten[s] = []
        y_ten[s] = []
        for k, v in d.items(): 
            x_ten[s].append(v['far'])
            y_ten[s].append(v['hr'])
    
    # aggregate plots of HR, FAR for first 10 subjects for each C
    for subj in x_ten.keys():
        xs, ys = x_ten[subj], y_ten[subj]
        plt.plot(xs, ys, '-o', label=subj, ms=5)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('ROC CURVES FOR FIRST 10 SUBJECTS')
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Hit Rate')
    plt.legend(title='Subject index', title_fontsize='small', fontsize='small')
    plt.show()
    
def q3(data):
    # average HR, FAR across all subjects for each C
    hrsf, farsf = q1b(data)[:-2]
    
    # plot average HR, FAR across subjects for each C
    plt.plot(farsf.values(), hrsf.values(), '-o', ms=5, color='purple')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('ROC CURVE AVERAGED ACROSS ALL SUBJECTS')
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Hit Rate')
    plt.show()
    
def q4(data):
    # hits and misses for each lag for each subject, C=4
    lags = dict()
    for idx, sdata in data.groupby('Subject'):
        relevant = sdata[sdata['Study-test lag'] > 0]
        lags[idx] = dict()
        for _, d in relevant.iterrows():
            lag = d['Study-test lag']
            if d['Confidence rating'] > 4:
                lags[idx].setdefault(lag, {'hits': 0, 'misses': 0})['hits'] += 1
            else:
                lags[idx].setdefault(lag, {'hits': 0, 'misses': 0})['misses'] += 1
    
    # HR for each lag for each subject
    # aggregate HR for each lag across subjects
    lags_data = dict()
    for idx, ldata in lags.items():
        for lag, d in ldata.items():
            lags_data.setdefault(lag, []).append(hitrate(d))
    
    # average HR for each lag across subjects 
    lagsf = dict()
    for lag, hrs in lags_data.items():
        lagsf[lag] = sum(hrs) / len(hrs)
    lagsf = dict(sorted(lagsf.items()))
    
    # plot HR against lag across subjects
    plt.plot(list(lagsf), list(lagsf.values()), '-o', ms=5, color='green')
    plt.xlim([0, 500])
    plt.ylim([0, 1])
    plt.title('HIT RATE AGAINST STUDY-TEST LAG')
    plt.xlabel('Hit Rate')
    plt.ylabel('Study-Test Lag')
    plt.show()
    
def q5(data):
    # reaction time for each hit, miss, FA, CR for each subject, C=4
    rt_data = dict()
    for idx, sdata in data.groupby('Subject'):
        res = {'hits': [], 'misses': [], 'fas': [], 'rejs': []}
        for _, row in sdata.iterrows():
            rt = row['Reaction time']
            cr = row['Confidence rating']
            if row['Study-test lag'] > 0:
                if cr > 4:
                    res['hits'].append(rt)
                else:
                    res['misses'].append(rt)
            else:
                if cr > 4:
                    res['fas'].append(rt)
                else:
                    res['rejs'].append(rt)
        rt_data[idx] = res
    
    # mean reaction time for hits, misses, FAs, CRs for each subject
    rts = dict()
    for idx, rdata in rt_data.items():
        res = dict()
        for stat, d in rdata.items():
            res[stat] = sum(d) / len(d)
        rts[idx] = res
        
    # aggregate mean reaction times for each stat across subjects
    rt_data = {'hits': [], 'misses': [], 'fas': [], 'rejs': []}
    for r in rts.values():
        for stat, v in r.items():
            rt_data[stat].append(v)
    
    # average reaction time for each stat across subjects
    rtsf = dict()
    for stat, v in rt_data.items():
        rtsf[stat] = sum(v) / len(v)
    
def q6(data):
    # reaction times for each lag for each subject, C=4
    lags = dict()
    for idx, sdata in data.groupby('Subject'):
        relevant = sdata[sdata['Study-test lag'] > 0]
        lags[idx] = dict()
        for _, d in relevant.iterrows():
            lag = d['Study-test lag']
            if d['Confidence rating'] > 4:
                lags[idx].setdefault(lag, []).append(d['Reaction time'])
    
    # mean reaction times for each lag for each subject
    slags = dict()            
    for s, d in lags.items():
        slags[s] = dict()
        for lag, rts in d.items():
            slags[s][lag] = sum(rts) / len(rts)
    
    # aggregate mean reaction times for each lag across subjects
    lags_data = dict()
    for d in slags.values():
        for lag, mean_rt in d.items():
            lags_data.setdefault(lag, []).append(mean_rt)
    
    # average reaction time for each lag across subjects
    lagrt = dict()
    for lag, mrt in lags_data.items():
        lagrt[lag] = sum(mrt) / len(mrt)
    lagrt = dict(sorted(lagrt.items()))
    
    # plot average reaction time against lag
    plt.plot(list(lagrt), list(lagrt.values()), '-o', ms=5)
    plt.xlim([0, 500])
    plt.ylim([0, 2200])
    plt.title('BETWEEN-SUBJECT AVERAGE REACTION TIME AGAINST STUDY-TEST LAG')
    plt.xlabel('Study-Test Lag')
    plt.ylabel('Average Reaction Time / ms')
    plt.show()
    