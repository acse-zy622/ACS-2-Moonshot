import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
from tqdm import tqdm

import preproc

def loadcsv(path):
    '''
    Read output CSV file and return list of 
    Lat/Long values of craters.
    
    Parameters
    --------------
    path: str
        CSV filepath
        
    Returns
    --------------
    list
    '''
    df = pd.read_csv(path, sep=" ")

    assert len(df.columns) == 5
    assert len(df.index) > 1

    dimens = df.columns.astype(np.float64).tolist() 

    df.iloc[:,2], df.iloc[:,1], df.iloc[:,3] = preproc.transcoor(
        df.iloc[:,2]*416, df.iloc[:,1]*416, 416, 416, dimens[3], dimens[4], 
        dimens[1], dimens[2], df.iloc[:,3]
    )

    return df.iloc[:,1:4].to_dict('split')['data']


def returnCrater(keywords=[""]):
    '''
    Generate dataframe containing all craters from 
    separate CSV files.
    
    Parameters
    -------------
    keywords: list
        read CSV file keyword
    
    Returns
    -------------
    pd.DataFrame
    '''
    csvPaths=[]

    for (dirpath,_,files) in os.walk(os.getcwd()):
        csvPaths += [os.sep.join((dirpath,filename)) 
                     for filename in files 
                     if (".csv" in filename.lower())
                     and
                     any(pattern.lower() in filename.lower() 
                         for pattern in keywords)
                    ]

    with ThreadPoolExecutor(max_workers=min(os.cpu_count()+4, len(csvPaths))) as executor:
        futures = []
        dfList=[]
        for path in tqdm(csvPaths):
            futures.append(executor.submit(loadcsv, path))

        done,_ = wait(futures)

        for future in done:
            try:
                dfList += future.result()
            except:
                continue
                
    return pd.DataFrame(dfList, columns=['Latitude', 'Longitude', 'Radius (km)'])
            
    
if __name__ == "__main__":
    craterdf = returnCrater()
    sns.kdeplot(craterdf['Radius (km)'], log_scale=(True,True), bw_adjust=5)