import pandas as pd
import numpy as np

def samp(dsn, stratas, sample_size):
    """
    parameters
    dsn: population dataset
    stratas: column nsmaes as list format ex: ['strata1','strata2','strata3....']
    sample_size: how much percentage you want as a sample like 0.05, 0.10, 0.25...
    """
    freq=dsn.groupby(stratas).size().reset_index(name='counts')
    freq['pct']=freq['counts']/dsn.shape[0]
    samp_size_num=(dsn.shape[0])*sample_size
    freq['sample']=(freq['pct']*samp_size_num).apply(np.round)
    freq=freq[freq['sample']>0] # if the sample is less than 0
    sample=pd.DataFrame()
    for index,row in freq.iterrows():
        freq_bkt=freq.loc[freq.index==index]
        result = pd.merge(dsn, freq_bkt, on=stratas)
        samp=result.sample(n=int(row['sample']), random_state=99)
        sample=sample.append(samp)
    sample.drop(['counts','pct', 'sample'], axis=1,inplace=True)
    return sample
