# **********************************
#   Author: Suraj Subramanian
#   2nd January 2020
# **********************************

import pandas as pd
import utils
import pickle
import numpy as np
import torch.utils.data as data
import random, os, sys
from sklearn.preprocessing import MinMaxScaler

def clean_longitudinal_inputs(load_pkl=True, padded=True, exclude_ct=True):
    if load_pkl:
        df = pd.read_pickle('pickles/du2/monthwise_inputs.pkl')
        col_dict = utils.pkl_load('pickles/du2/col_dict.pkl')
        x_mean = utils.pkl_load('pickles/du2/x_mean.pkl')
        return df, col_dict, x_mean

    df = load_irregular_sequence(exclude_ct)    
    del df['ENC_PK_INJ']
    df = apply_monthly_timestamp(df)

    # Empirical mean
    x_mean = Imputer(df).commonsense_imputer().mean()
    x_mean['HBI_UC_SCORE'] = df.HBI_UC_SCORE.quantile(0.5)
    x_mean['HBI_CROHNS_SCORE'] = df.HBI_CROHNS_SCORE.quantile(0.5)
    utils.pkl_dump(MinMaxScaler((-0.5, 0.5)).fit_transform(x_mean.values.reshape(1,-1)), 'pickles/du2/x_mean.pkl')

    df = pad_missing_months(df)

    # Impute constant ffill for non-dynamic variables
    df['DS_CD'] = df.DS_CD.fillna(None, 'ffill')
    df['DS_UC'] = df.DS_UC.fillna(None, 'ffill')
    df['DS_AGE_DX'] = df.DS_AGE_DX.fillna(None, 'ffill')
    df['DS_PREV_RESECTION'] = df.DS_PREV_RESECTION.fillna(None, 'ffill')
    
    m, t = missingness_indicators(df)
    
    fit_scaler(pd.concat([df,m,t], axis=1))

    df = Imputer(df).clinical_impute() # Impute acc to clinical rules
    print("Imputation done")
    
    col_dict = {'input':df.columns, 'missing':m.columns, 'delta':t.columns}
    df.to_pickle('pickles/du2/x_padded_inputs.pkl')
    m.to_pickle('pickles/du2/m_missing_mask.pkl')
    t.to_pickle('pickles/du2/t_missing_delta.pkl')
    all_data = pd.concat([df, m, t], axis=1)
    all_data.to_pickle('pickles/du2/monthwise_inputs.pkl')  
    with open('pickles/du2/col_dict.pkl', 'wb') as f:
        pickle.dump(col_dict, f)
    
    print("All data saved in pickles/du2/")
    
def load_irregular_sequence(exclude_ct=True):
    """
    Loading irregular sequence from disk
    """
    print(load_irregular_sequence.__doc__)

    if exclude_ct:
        return pd.read_csv(os.path.join('preprocessing','final', 'noCT.csv'))
    return pd.read_csv(os.path.join('preprocessing','final', 'all.csv'))

def apply_monthly_timestamp(df): 
    """
    Applying a monthly timestamp to each observation. Each patient's first observation starts with 0. 
    Subsequent observations have a timestamp denoting the number of months since first observation
    """
    print(apply_monthly_timestamp.__doc__)

    df['tmp_ts'] = (df.RESULT_YEAR.astype(str)+df.RESULT_MONTH.astype(str).str.zfill(2)).astype(int)
    t0_indices = df.groupby('AUTO_ID').tmp_ts.idxmin()     # find earliest row
    t0_df = df.loc[t0_indices][['AUTO_ID','RESULT_YEAR','RESULT_MONTH']]
    t0_df.columns = ['AUTO_ID', 'tmp_minY', 'tmp_minMo']
    # add tmp columns to df for vectorized calculation of delta in months
    timestamped_df = df.merge(t0_df, how='left', on='AUTO_ID')
    timestamped_df['MONTH_TS'] = (timestamped_df['RESULT_YEAR']-timestamped_df['tmp_minY'])*12 + timestamped_df['RESULT_MONTH']-timestamped_df['tmp_minMo']
    df = timestamped_df[[x for x in timestamped_df.columns if x[:3]!='tmp']]
    print(f"Data shape: {df.shape}")
    return df

def pad_missing_months(timestamped_df):
    """
    Resampling the data to a monthly rate.
    """
    print(pad_missing_months.__doc__)

    def pad(df):
        df = df.reset_index()
        new_index = pd.MultiIndex.from_product([[df.AUTO_ID.values[0]], list(range(0, df.MONTH_TS.max()+1))], names=['AUTO_ID', 'MONTH_TS'])
        df = df.set_index(['AUTO_ID', 'MONTH_TS']).reindex(new_index).reset_index()
        return df
    df = timestamped_df.groupby('AUTO_ID', as_index=False).apply(pad)
    df.index = range(0, df.shape[0])
    df.drop(['index', 'RESULT_YEAR', 'RESULT_MONTH'], axis=1, inplace=True)
    print(f"Data shape: {df.shape}")
    return df

def fit_scaler(df):
    """
    Fitt scaler to normalize variables in a range of -0.5 to +0.5
    """
    print(fit_scaler.__doc__)
    scaler = MinMaxScaler((-0.5, 0.5)).fit(df)
    utils.pkl_dump(scaler, 'pickles/du2/input_scaler.pkl')
    print("Saved fitted scaler to pickles/du2/input_scaler.pkl")


def missingness_indicators(padded):
    """
    Computing the `missing_mask` and `missing_delta` matrices. 
    missing_mask is 1 if variable is observed, 0 if not.
    missing_delta tracks the number of `time_interval`s (here, 1 month) between two observations of a variable
    """
    print(missingness_indicators.__doc__)

    missing_mask = padded.copy()
    missing_mask.loc[:] = missing_mask.loc[:].notnull().astype(int)
    missing_mask['AUTO_ID'] = padded['AUTO_ID'].copy() # restore auto_ids
    missing_delta = missing_mask.copy()
    
    time_interval = 1 
    missing_delta['tmp_delta'] = time_interval
    for observed in missing_delta.columns[1:]:
        missing_delta['obs_delays'] = missing_delta.groupby('AUTO_ID')[observed].cumsum()
        missing_delta[observed] = missing_delta.groupby(['AUTO_ID', 'obs_delays'])['tmp_delta'].cumsum() -1
    missing_delta = missing_delta.drop(['tmp_delta', 'obs_delays'], axis=1)

    missing_mask.columns = ['AUTO_ID']+[f'MISSING_{x}' for x in padded.columns[1:]]
    missing_delta.columns = ['AUTO_ID']+[f'DELTA_{x}' for x in  padded.columns[1:]]
    return missing_mask.drop(['AUTO_ID'],axis=1), missing_delta.drop(['AUTO_ID'],axis=1)


                       



class Imputer:
    def __init__(self, df=None):
        self.df = df

        self.cols_to_impute = ['ENC_OFF_Related', 'ENC_OFF_Unrelated',
       'ENC_PROC_Related', 'ENC_PROC_Unrelated', 'ENC_TEL_Related',
       'ENC_TEL_Unrelated', 'ENC_SURGERY', 'DIAG_COLONOSCOPY',
       'DIAG_ENDOSCOPY', 'DIAG_SIGMOIDOSCOPY', 'DIAG_ILEOSCOPY', 'DIAG_ANO',
       'DIAG_CT_ABPEL', 'LAB_albumin_High', 'LAB_albumin_Low',
       'LAB_albumin_Normal', 'LAB_crp_High', 'LAB_crp_Low', 'LAB_crp_Normal',
       'LAB_eos_High', 'LAB_eos_Low', 'LAB_eos_Normal', 'LAB_esr_High',
       'LAB_esr_Normal', 'LAB_hemoglobin_High', 'LAB_hemoglobin_Low',
       'LAB_hemoglobin_Normal', 'LAB_monocytes_High', 'LAB_monocytes_Low',
       'LAB_monocytes_Normal', 'LAB_vitamin_d_High', 'LAB_vitamin_d_Low',
       'LAB_vitamin_d_Normal', 'MED_5_ASA', 'MED_Systemic_steroids',
       'MED_Immunomodulators', 'MED_Psych', 'MED_Vitamin_D', 'MED_ANTI_TNF',
       'MED_ANTI_IL12', 'MED_ANTI_INTEGRIN', 'HBI_CROHNS_SCORE',
       'HBI_UC_SCORE', 'DS_CD', 'DS_UC', 'DS_AGE_DX', 'DS_PREV_RESECTION']

        self.enc_diag_rx_cols = [x for x in self.cols_to_impute if x[:3]=='ENC' or x[:4]=='DIAG' or x[:3]=='MED']
        self.hbi_cols = [x for x in self.cols_to_impute if x[:3]=='HBI']
        self.lab_fillna = {f'LAB_{g}_{l}':(1 if l=='Normal' else 0) \
                            for g in ['albumin', 'eos', 'hemoglobin', 'monocytes', 'vitamin_d', 'crp', 'esr'] \
                                for l in ['High', 'Low', 'Normal']}
    
    def clinical_impute(self):
        """
        Clinical Imputation:
         - Use ffill for LABS, HBI Scores
         - Use commonsense for DIAG, ENC,  MEDS (if it isn't recorded, it didn't happen -> impute 0)
        """
        print(Imputer.clinical_impute.__doc__)
        df = self.df.copy()
        df.loc[:, self.enc_diag_rx_cols] = df.loc[:, self.enc_diag_rx_cols].fillna(0) # commonsense
        df = df.groupby('AUTO_ID').apply(self.forward) # ffill
        return df

    def forward(self, grp):
        # fillna of 0th row with commonsense values    
        grp.loc[grp.MONTH_TS==0,:] = self.commonsense_imputer(grp.loc[grp.MONTH_TS==0,:])
        grp.loc[:,self.cols_to_impute] = grp.loc[:,self.cols_to_impute].fillna(None, 'ffill')
        return grp.fillna(0)

    def commonsense_imputer(self, df=None):
        """
        - 0 for encounters (if it wasn't recorded, it probably didn't happen)
        - 0 for diagnoses (-- " --)
        - 0 for RX (-- " --)
        - 1 for LAB_*_Normal (if it wasn't prescribed, it's probably normal)
        - -1 for HBI (wasn't conducted)
        """
        if df is None:
            df = self.df.copy()
        # impute encounter, diag, meds
        df.loc[:, self.enc_diag_rx_cols] = df.loc[:, self.enc_diag_rx_cols].fillna(0)
        # impute labs
        df = df.fillna(self.lab_fillna)
        # impute HBI
        df.loc[:,self.hbi_cols] = df.loc[:,self.hbi_cols].fillna(-1)
        return df

    
    


class TargetFunc:
    def __init__(self, data):
        self.data = data


    def annual_charges(self, indices, thresholds=[10000, 80000]): # 3 target classes
        df = self.data.loc[indices[0]:indices[1]]
        y = df.RELATED_IP.fillna(0) #+ df.RELATED_OP.fillna(0)
        annualized = y.mean()*12
        thresholds.append(sys.maxsize)
        for c,t in enumerate(thresholds):
            if annualized<=t:
                return c
    
    
    def abnormal_labs(self, indices):
        cols = ['LAB_albumin_Low', 'LAB_crp_High', 'LAB_eos_High', 'LAB_esr_High', \
            'LAB_hemoglobin_Low', 'LAB_monocytes_High', 'LAB_monocytes_Low', 'LAB_vitamin_d_Low']
        # Mark 1 if even one occurrence, else 0
        df = self.data.loc[indices[0]:indices[1]]
        y = df[cols].fillna(0).sum().values
        y = np.minimum(np.ones_like(y), y)
        return y

    def surgery(self, indices):
        # Mark 1 if even one occurrence, else 0
        df = self.data.loc[indices[0]:indices[1]]
        return min(1, df.ENC_SURGERY.fillna(0).sum())
    
    def diagnostics(self, indices):
        cols = ['DIAG_COLONOSCOPY',
                'DIAG_ENDOSCOPY',
                'DIAG_SIGMOIDOSCOPY',
                'DIAG_ILEOSCOPY',
                'DIAG_ANO',
                'DIAG_CT_ABPEL']
        df = self.data.loc[indices[0]:indices[1]]
        # Mark 1 if even one occurrence, else 0
        y = df[cols].fillna(0).sum().values
        y = np.minimum(np.ones_like(y), y)
        return y


   
        
class IBDSequenceGenerator:
    
    def __init__(self, history_len, future_len):
        """
        Patient Sequence-Label generator
        history_len: length of training sequence (default=24 months)
        future_len: length of future sequence to predict on (default=8 months)
        """
        self.history_len = history_len
        self.future_len = future_len
        self.min_rows_reqd=self.history_len+self.future_len
        self.col_dict=None
        self.scaler = utils.pkl_load('pickles/du2/input_scaler.pkl')

   
    def df_to_training_pairs(self, seq_dir, auto_id, grp):
       
        def decompose_slice(indices):
            try:
                input_df = grp.loc[indices[0]:indices[1]] # Slice out an input sequence
                first_month = input_df.MONTH_TS.iloc[0]
            except:
                print(grp.index, indices)
            name = f'{auto_id}_{first_month}'

            input_df = pd.DataFrame(self.scaler.transform(input_df), columns=input_df.columns)
            x = input_df[self.col_dict['input']] 
            x_obs = x.shift(1).fillna(x.iloc[0]).values
            x = x.values
            m = input_df[self.col_dict['missing']].values
            t = input_df[self.col_dict['delta']].values
            
            return name, np.array([x[:,2:], m[:,1:], t[:,1:], x_obs[:,2:]]) # remove auto id and month_ts

        input_start_ix = grp.index[0]
        input_final_ix = grp.index[-1]-self.min_rows_reqd+1
        input_seq_df = [(pos, pos + self.history_len-1) for pos in range(input_start_ix, input_final_ix+1)] # pandas slicing is inclusive

        target_start_ix = grp.index[0] + self.history_len
        target_final_ix = grp.index[-1]-self.future_len+1
        target_seq_df = [(pos, pos+self.future_len-1) for pos in range(target_start_ix, target_final_ix+1)]
        tgt_types = ['annual_charges', 'abnormal_labs', 'diagnostics', 'surgery']
        label_dict={}
        TargetLoader = TargetFunc(grp)

        # sliding window
        for slice_ix in range(len(input_seq_df)):
            name, x = decompose_slice(input_seq_df[slice_ix])
            utils.pkl_dump(x, os.path.join(seq_dir, name+'.npy'))
            
            label_dict[name] = {}
            for tgt in tgt_types:
                fn = eval(f"TargetLoader.{tgt}")
                try:
                    label_dict[name][tgt] = fn(target_seq_df[slice_ix])
                except:
                    print(grp.index, TargetLoader.data.index, target_seq_df[slice_ix])
        
        return label_dict # return for collation with other patients




    
    def write_to_disk(self, datadir, train_size):
        """
        IBDSequenceGenerator.write_to_disk:
         - Reads padded X, M, T matrices
         - Impute according to clinical rules
         - Generate overlapping sequences and {tgt_type} training labels for each patient
         - Scale input sequences
         - Persist sequences as numpy arrays at data/{datadir}/{SEQ_DIR}/{AUTO_ID}_{START_TS}.npy
         - Persist labels for all sequences in data/{datadir}/{SEQ_DIR}/label_dict.pkl
         - Persist empirical mean values at data/{datadir}/x_mean.pkl
        """
        print(IBDSequenceGenerator.write_to_disk.__doc__)

        if not datadir: datadir=f'm{self.history_len}p{self.future_len}'
        out_fp = os.path.join('data', datadir)
        train_seq_dir = os.path.join(out_fp, 'train')
        valid_seq_dir = os.path.join(out_fp, 'valid')
        test_seq_dir = os.path.join(out_fp, 'test')
        if not os.path.exists(out_fp): os.makedirs(out_fp)
        if not os.path.exists(train_seq_dir): os.makedirs(train_seq_dir)
        if not os.path.exists(valid_seq_dir): os.makedirs(valid_seq_dir)
        if not os.path.exists(test_seq_dir): os.makedirs(test_seq_dir)
                
        dataset, self.col_dict, x_mean = clean_longitudinal_inputs(load_pkl=True) # Load inputs

        print("Inputs loaded")
    
        groups = dataset.groupby('AUTO_ID') # 2863 patients

        TRAIN_LABELS={}
        VALID_LABELS = {}
        TEST_LABELS = {}

        valid_size = train_size + (1-train_size)/2
        patients_dropped = 0
        for auto_id, grp in groups:
            grplen = grp.shape[0]
            if grplen < self.min_rows_reqd: 
                patients_dropped+=1
                continue

            train = grp.iloc[:int(train_size*grplen)]
            # print("train", train.index)
            valid = grp.iloc[int(train_size*grplen):int(valid_size*grplen)]
            # print("valid", valid.index)
            test = grp.iloc[int(valid_size*grplen):]
            # print("train", test.index)

            if valid.shape[0] < self.min_rows_reqd: #1415 patients (for m7p3)
                if valid.shape[0]+test.shape[0] >= self.min_rows_reqd: #898
                    test = pd.concat([valid,test])
                    # print(test.index)
                    valid = None
                else: # 517
                    train = pd.concat([train, valid, test])
                    # print(train.index)
                    valid = None
                    test = None
            
            TRAIN_LABELS.update(self.df_to_training_pairs(train_seq_dir, auto_id, train))
            if valid is not None:
                VALID_LABELS.update(self.df_to_training_pairs(valid_seq_dir, auto_id, valid))
            if test is not None:
                TEST_LABELS.update(self.df_to_training_pairs(test_seq_dir, auto_id, test))

        utils.pkl_dump(TRAIN_LABELS, os.path.join(train_seq_dir, 'label_dict.pkl'))
        utils.pkl_dump(VALID_LABELS, os.path.join(valid_seq_dir, 'label_dict.pkl'))
        utils.pkl_dump(TEST_LABELS, os.path.join(test_seq_dir, 'label_dict.pkl'))
        
        utils.pkl_dump(x_mean[2:], os.path.join(out_fp, 'x_mean.pkl')) # remove auto id and month ts
        print(f"PAtients dropped: {patients_dropped}")




class PatientDataset2(data.Dataset):  
    def __init__(self, datadir, type): 
        self.datadir = os.path.join('data',datadir, type) # will contain train/test/valid and {tr,v,t}_label_dict.pkl
        self.label_dict = utils.pkl_load(os.path.join(self.datadir,'label_dict.pkl'))
        self.list_IDs = list(self.label_dict.keys())
        

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = utils.pkl_load(os.path.join(self.datadir, ID+'.npy')) # Numpy seq
        y = self.label_dict[ID] # Dict of available target values
        return X, y

"""
ARCHIVE
- Overlapping split code


class PatientDataset(data.Dataset):  
    def __init__(self, datadir, list_IDs): 
        self.datadir = os.path.join('data',datadir)
        self.list_IDs = list_IDs
        self.all_labels = {}
        avbl_dicts = {x.replace('_label_dict.pkl', '') : utils.pkl_load(os.path.join(self.datadir,x)) \
                        for x in os.listdir(self.datadir) if '_label_dict.pkl' in x} # {tgt_type : tgt_dict}
                        
        for ID in self.list_IDs:
            self.all_labels[ID] ={tgt_type:tgt_dict[ID] for tgt_type, tgt_dict in avbl_dicts.items()}

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = utils.pkl_load(os.path.join(self.datadir, 'sequences', ID+'.npy')) # Numpy seq
        y = self.all_labels[ID] # Dict of available target values
        return X, y  


def sequence_split(datadir, dataset_name, train_size=0.8, stratify_by=None, holdout_patients=['2544', '3046', '1359', '1335', '2298', '785']):
    """
    # Populate Tr/V/Te Datasets with {datadir}/{sequences}/*.npy and {datadir}/*_label_dict.pkl
    # Split will be stratified according to labels in {datadir}/{stratify_by}_label_dict.pkl \
    #     else according to first label_dict found.
    """
    from sklearn.model_selection import train_test_split

    if stratify_by is not None:
        # Load target-dictionary to stratify our splits by
        label_dict = utils.pkl_load(os.path.join('data', datadir, stratify_by+'_label_dict.pkl'))
    else:
        # If not specified, simple stratify by the first available target-dictionary
        avbl_tgts = [x for x in os.listdir(datadir) if '_label_dict.pkl' in x]
        label_dict = utils.pkl_load(os.path.join('data', datadir, avbl_tgts[0]))
        stratify_by = avbl_tgts[0]
    
    holdout_ids = [] # We will add this to the test set
    x = []
    y = []
    for x_, y_ in label_dict.items():
        if x_.split('_')[0] in holdout_patients:
            holdout_ids.append(x_)
        else:
            x.append(x_)
            y.append(y_)
       
    train, valid = train_test_split(x, train_size=train_size, stratify=y)
    if holdout_ids:
        test = holdout_ids
    else:
        valid, test = train_test_split(valid, train_size=0.5)    
    

    meta = {'stratify_by':stratify_by, 
            'train_labels':pd.Series(map(lambda x:label_dict[x], train)).value_counts().values,
            'valid_labels':pd.Series(map(lambda x:label_dict[x], valid)).value_counts().values,
            'test_labels':pd.Series(map(lambda x:label_dict[x], test)).value_counts().values,
        }
    print(meta)

    # Create the dataset pickles for passing to model
    build_dataset(datadir, dataset_name, train, valid, test, meta)

    
def interpatient_split(datadir, dataset_name, train_size=0.8, stratify_by=None):
    """
    # Populate Tr/V/Te Datasets with {datadir}/{sequences}/*.npy and {datadir}/*_label_dict.pkl
    # Split will be stratified according to labels in {datadir}/{stratify_by}_label_dict.pkl else according to any dict
    """
    from collections import defaultdict
    from sklearn.model_selection import train_test_split

    label_dict = utils.pkl_load(os.path.join('data', datadir, 'annual_charges_label_dict.pkl'))
    pat_seq_dict = defaultdict(list)
    for k in label_dict.keys():
        pat_seq_dict[k.split('_')[0]].append(k)
    
    patients = set(pat_seq_dict.keys())
    reqd_test_pats = ['2544', '3046', '1359', '1335', '2298', '785']
    patients = list(patients - set(reqd_test_pats))
    
    train_p, test_p = train_test_split(patients, train_size=train_size)
    valid_p, test_p = train_test_split(test_p, train_size=0.5)  
    test_p.extend(reqd_test_pats)

    train = []
    for p in train_p: train.extend(pat_seq_dict[p])
  

    valid = []
    for p in valid_p: valid.extend(pat_seq_dict[p])

    test = []
    for p in test_p: test.extend(pat_seq_dict[p])

    meta = {
        'train_labels':pd.Series(map(lambda x:label_dict[x], train)).value_counts().values,
        'valid_labels':pd.Series(map(lambda x:label_dict[x], valid)).value_counts().values,
        'test_labels':pd.Series(map(lambda x:label_dict[x], test)).value_counts().values,
    }
    print(meta)
    # Create the dataset pickles for passing to model
    build_dataset(datadir, dataset_name, train, valid, test, meta)


def intrapatient_split(datadir, dataset_name, train_size=0.8, holdout_patients=['2544', '3046', '1359', '1335', '2298', '785']):
    """
    # Populate Tr/V/Te Datasets with {datadir}/{sequences}/*.npy and {datadir}/*_label_dict.pkl
    # Split will be stratified according to labels in {datadir}/{stratify_by}_label_dict.pkl else according to any dict
    """
    from collections import defaultdict
    valid_size = train_size + (1-train_size)/2
    label_dict = utils.pkl_load(os.path.join('data', datadir, 'annual_charges_label_dict.pkl'))

    print("Grouping sequences")
    # Group sequences by patient ID
    pat_seq_dict = defaultdict(list)
    for k in label_dict.keys():
        pat_seq_dict[k.split('_')[0]].append(k)
    print("Grouping done")

    # Get TVT split on each patient
    train=[]
    valid=[]
    test=[]
    TrV_overlap=[]
    TrT_overlap = []

    for patient_sequences in pat_seq_dict.values():
        # Split each patient's sequences based on time.
        Tr = patient_sequences[:round(train_size*len(patient_sequences))]
        V = patient_sequences[round(train_size*len(patient_sequences)) : round(valid_size*len(patient_sequences))]
        T = patient_sequences[round(valid_size*len(patient_sequences)):]
        
        train.extend(Tr)
        valid.extend(V)
        test.extend(T)

        if len(T)>1:
            TrV_overlap.append((int(Tr[-1].split('_')[1])+24 - int(V[0].split('_')[1]))/(24*len(V)))
            TrT_overlap.append((int(Tr[-1].split('_')[1])+24 - int(T[0].split('_')[1]))/(24*len(T)))
        
    
    meta = {
        'train_labels':pd.Series(map(lambda x:label_dict[x], train)).value_counts().values,
        'valid_labels':pd.Series(map(lambda x:label_dict[x], valid)).value_counts().values,
        'test_labels':pd.Series(map(lambda x:label_dict[x], test)).value_counts().values,
        'train_valid_overlap': pd.Series(TrV_overlap).describe(),
        'train_test_overlap': pd.Series(TrT_overlap).describe()
    }

    # Create the dataset pickles for passing to model
    build_dataset(datadir, dataset_name, train, valid, test, meta)


def build_dataset(datadir, dataset_name, train, valid, test, meta):
    from shutil import copyfile
    tr, v, te = PatientDataset(datadir, train), PatientDataset(datadir, valid), PatientDataset(datadir, test) 
    os.makedirs(os.path.join('datasets', dataset_name))
    utils.pkl_dump(meta, os.path.join('datasets', dataset_name,'meta.data'))
    utils.pkl_dump(tr, os.path.join('datasets', dataset_name,'train.dataset'))
    utils.pkl_dump(v, os.path.join('datasets', dataset_name,'valid.dataset'))
    utils.pkl_dump(te, os.path.join('datasets', dataset_name,'test.dataset'))
    copyfile(os.path.join('data', datadir, 'x_mean.pkl'), os.path.join('datasets', dataset_name,'x_mean.data'))
    print(f"Dataset written at datasets/{dataset_name}")

"""