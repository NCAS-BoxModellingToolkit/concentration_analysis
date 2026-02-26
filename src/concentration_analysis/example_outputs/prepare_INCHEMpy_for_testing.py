import numpy as np
import pandas as pd
def main():

    out_data={}
    i = '/Users/user/Library/CloudStorage/OneDrive-TheUniversityofManchester/GitHub/NCAS-BoxModellingToolkit/concentration_analysis/src/concentration_analysis/example_outputs/fig09/INCHEM-photoout'
    with open("%s/out_data.pickle" % i,'rb') as handle:
        out_data['test']=(pd.read_pickle(handle))
    
    # get times
    t = np.array((out_data['test'].index.to_list()))
    tindx = t<7000.

    # get column headers
    ch = out_data['test'].columns.values

    gindx = np.zeros((len(ch)))==1
    gindx[ch == 'O3'] = True
    gindx[ch == 'O'] = True
    gindx[ch == 'O1D'] = True
    gindx[ch == 'NO'] = True
    gindx[ch == 'NO2'] = True

    data = {"O3": out_data['test'].loc[tindx,"O3"],
            "O": out_data['test'].loc[tindx,"O"],
            "O1D": out_data['test'].loc[tindx,"O1D"],
            "NO": out_data['test'].loc[tindx,"NO"],
            "NO2": out_data['test'].loc[tindx,"NO2"]}

    df = pd.DataFrame(data)
    tstr = []
    for i in range(len(t)):
        if float(t[i]) < 7000:
            tstr.append(str(t[i]))
    
    df.index = tstr

    df.to_pickle('/Users/user/Library/CloudStorage/OneDrive-TheUniversityofManchester/GitHub/NCAS-BoxModellingToolkit/concentration_analysis/src/concentration_analysis/example_outputs/for_testing/out_data.pickle')

    return()

main()