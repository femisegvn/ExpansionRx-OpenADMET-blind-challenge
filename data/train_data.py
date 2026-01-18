import numpy as np
import pandas as pd
from io import StringIO

train_df = pd.read_csv("openadmet-expansionrx-challenge-train-data/expansion_data_train.csv")

data = """Assay,Log_Scale,Multiplier,Log_name
LogD,False,1,LogD
KSOL,True,1e-6,LogS
HLM CLint,True,1,Log_HLM_CLint
MLM CLint,True,1,Log_MLM_CLint
Caco-2 Permeability Papp A>B,True,1e-6,Log_Caco_Papp_AB
Caco-2 Permeability Efflux,True,1,Log_Caco_ER
MPPB,True,1,Log_Mouse_PPB
MBPB,True,1,Log_Mouse_BPB
MGMB,True,1,Log_Mouse_MPB
"""

s = StringIO(data)

conversion_df = pd.read_csv(s)

conversion_dict = dict([(x[0], x[1:]) for x in conversion_df.values])

log_train_df = train_df[["SMILES", "Molecule Name"]].copy()

for col in train_df.columns[2:]:
    log_scale, multiplier, short_name = conversion_dict[col]
    
    log_train_df[short_name] = train_df[col].astype(float)
    if log_scale:
    
        log_train_df[short_name] = log_train_df[short_name] + 1
    
        log_train_df[short_name] = np.log10(log_train_df[short_name] * multiplier)

log_col_names = log_train_df.columns[2:]