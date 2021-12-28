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
        df = pd.read_pickle('pickles/du1/monthwise_inputs.pkl')
        col_dict = utils.pkl_load('pickles/du1/col_dict.pkl')
        x_mean = utils.pkl_load('pickles/du1/x_mean.pkl')
        return df, col_dict, x_mean

    df = load_irregular_sequence(exclude_ct)    
    del df['ENC_PK_INJ']
    df = apply_monthly_timestamp(df)
    df = pad_missing_months(df)

    # Impute constant ffill for non-dynamic variables
    df['DS_CD'] = df.DS_CD.fillna(None, 'ffill')
    df['DS_UC'] = df.DS_UC.fillna(None, 'ffill')
    df['DS_AGE_DX'] = df.DS_AGE_DX.fillna(None, 'ffill')
    df['DS_PREV_RESECTION'] = df.DS_PREV_RESECTION.fillna(None, 'ffill')

    fit_scaler(df)
    
    # Empirical mean
    x_mean = Imputer(df).commonsense_imputer().mean()
    x_mean['HBI_UC_SCORE'] = df.HBI_UC_SCORE.quantile(0.5)
    x_mean['HBI_CROHNS_SCORE'] = df.HBI_CROHNS_SCORE.quantile(0.5)
    x_mean.to_pickle('pickles/du1/x_mean.pkl')
    

    m, t = missingness_indicators(df)

    df = Imputer(df).clinical_impute() # Impute acc to clinical rules
    print("Imputation done")
    
    col_dict = {'input':df.columns, 'missing':m.columns, 'delta':t.columns}
    df.to_pickle('pickles/du1/x_padded_inputs.pkl')
    m.to_pickle('pickles/du1/m_missing_mask.pkl')
    t.to_pickle('pickles/du1/t_missing_delta.pkl')
    all_data = pd.concat([df, m, t], axis=1)
    all_data.to_pickle('pickles/du1/monthwise_inputs.pkl')  
    with open('pickles/du1/col_dict.pkl', 'wb') as f:
        pickle.dump(col_dict, f)
    
    print("All data saved in pickles/du1/")
    
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
    utils.pkl_dump(scaler, 'pickles/du1/input_scaler.pkl')
    print("Saved fitted scaler to pickles/du1/input_scaler.pkl")


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
        y = df.RELATED_OP.fillna(0) + df.RELATED_IP.fillna(0)
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


    # def _init_spikes(self, df, std_diff=1, min_charge=1000):
    #     """ tags charge as spike if >min_charge and std_diff std deviations away from median"""
    #     def find_spikes(charge, std_diff):
    #         mean = charge.mean()
    #         std = charge.std()
    #         scaled = (charge-mean)/std
    #         if std==0: 
    #             return pd.Series([0]*len(charge), index=charge.index)
    #         threshold = (min_charge-mean)/std
    #         median = scaled.quantile(0.5)
    #         spikes = np.where(scaled > max(threshold, (median+std_diff)), 1, 0) 
    #         return pd.Series(spikes, index=charge.index)

    #     df['charges'] = df.RELATED_IP.fillna(0) + df.RELATED_OP.fillna(0)
    #     df['spikes'] = df.groupby('AUTO_ID').charges.apply(find_spikes, std_diff=std_diff)
    #     df['spike_val'] = df['spikes'] * df['charges']
    #     self.spikes_df = df


    # def spikes(self, df, threshold=1):
    #     """ Threshold chosen from plotting spike distribution"""
    #     total_spikes = self.spikes_df.loc[df.index[0]:df.index[-1], 'spikes'].sum()
    #     return 1 if total_spikes > threshold else 0




   
        
class IBDSequenceGenerator:
    
    def __init__(self, history_len = 24, future_len = 8):
        """
        Patient Sequence-Label generator
        history_len: length of training sequence (default=24 months)
        future_len: length of future sequence to predict on (default=8 months)
        """
        self.history_len = history_len
        self.future_len = future_len
        self.min_rows_reqd=self.history_len+self.future_len
        self.col_dict=None
        self.scaler = utils.pkl_load('pickles/du1/input_scaler.pkl')

    
    def process_features(self, grp):
        """
        Input: Single Patient data
        Output: List of overlapping sequences reshaped for GRU
        """     
        def decompose_inputs(indices):
            input_df = grp.loc[indices[0]:indices[1]] # Slice out an input sequence
            x = input_df[self.col_dict['input']] 
            x_obs = self.scaler.transform(x.shift(1).fillna(x.iloc[0])) # Scale X_obs
            x = self.scaler.transform(x) # Scale X
            # M and T don't need scaling
            m = input_df[self.col_dict['missing']].values
            t = input_df[self.col_dict['delta']].values
            return np.array([x[:,1:], m, t, x_obs[:,1:]]) 

        input_start_ix = grp.index[0]
        input_final_ix = grp.index[-1]-self.min_rows_reqd+1
        input_seq_df = [(pos, pos + self.history_len-1) for pos in range(input_start_ix, input_final_ix+1)] # pandas slicing is inclusive
        sequences = np.array(list(map(decompose_inputs, input_seq_df)))
        return sequences


    def process_labels(self, grp, target_func):
        target_start_ix = grp.index[0] + self.history_len
        target_final_ix = grp.index[-1]-self.future_len+1
        target_seq_df = [(pos, pos+self.future_len-1) for pos in range(target_start_ix, target_final_ix+1)] # Get 'blind' slices
        labels = list(map(target_func, target_seq_df)) # Obtain label for each blind slice
        return labels

    
    def write_to_disk(self, datadir=None, tgt_types=None, labels_only=False, seq_only=False):
        """
        IBDSequenceGenerator.write_to_disk:
         - Reads padded X, M, T matrices
         - Impute according to clinical rules
         - Generate overlapping sequences and {tgt_type} training labels for each patient
         - Scale input sequences
         - Persist sequences as numpy arrays at data/{datadir}/sequences/{AUTO_ID}_{START_TS}.npy
         - Persist labels for all sequences in data/{datadir}/{tgt_type}_label_dict.pkl
         - Persist empirical mean values at data/{datadir}/x_mean.pkl
        """
        print(IBDSequenceGenerator.write_to_disk.__doc__)

        if not datadir:
            datadir=f'm{self.history_len}p{self.future_len}'
        out_fp = os.path.join('data', datadir)
        if not os.path.exists(out_fp): os.makedirs(out_fp)
        dataset, self.col_dict, x_mean = clean_longitudinal_inputs(load_pkl=True) # Load inputs
        print("Inputs loaded")
    

        if tgt_types is None:
            tgt_types = ['annual_charges', 'abnormal_labs', 'diagnostics', 'surgery']
        print(f"Generating labels for {tgt_types}")
        TargetLoader = TargetFunc(dataset[self.col_dict['input']]) # Pass input data for unscaling
        label_dicts = [{} for x in range(len(tgt_types))]
        
        groups = dataset.groupby('AUTO_ID') 
        for auto_id, grp in groups:
            # Skip patients who have lesser number of months 
            if grp.shape[0]<self.min_rows_reqd: 
                continue 

            # Save sequences
            if not labels_only:
                sequences = self.process_features(grp)
                seq_dir = os.path.join(out_fp, 'sequences')
                if not os.path.exists(seq_dir): os.makedirs(seq_dir)
                for c,x in enumerate(sequences):
                    ID = f'{auto_id}_{c}'
                    utils.pkl_dump(x, os.path.join(seq_dir, ID+'.npy'))

            # Get labels
            if not seq_only:
                for ix, tgt in enumerate(tgt_types):
                    func = eval(f"TargetLoader.{tgt}")
                    labels = self.process_labels(grp, func)
                    for c in range(len(labels)):
                        label_dicts[ix][f'{auto_id}_{c}'] = (labels[c])
        # Save labels
        for ix, tgt in enumerate(tgt_types):
            utils.pkl_dump(label_dicts[ix], os.path.join(out_fp, f'{tgt}_label_dict.pkl'))
        
        # Scale and save empirical mean without AUTO_ID
        scaled_mean = self.scaler.transform(x_mean.values.reshape(1,-1))
        scaled_mean = scaled_mean[:,1:]
        utils.pkl_dump(scaled_mean, os.path.join(out_fp, 'x_mean.pkl'))




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
    Populate Tr/V/Te Datasets with {datadir}/{sequences}/*.npy and {datadir}/*_label_dict.pkl
    Split will be stratified according to labels in {datadir}/{stratify_by}_label_dict.pkl \
        else according to first label_dict found.
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
    Populate Tr/V/Te Datasets with {datadir}/{sequences}/*.npy and {datadir}/*_label_dict.pkl
    Split will be stratified according to labels in {datadir}/{stratify_by}_label_dict.pkl else according to any dict
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


def intrapatient_split(datadir, dataset_name, train_size=0.8):
    """
    Populate Tr/V/Te Datasets with {datadir}/{sequences}/*.npy and {datadir}/*_label_dict.pkl
    Split will be stratified according to labels in {datadir}/{stratify_by}_label_dict.pkl else according to any dict
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
    print(meta)
    """
    {'train_labels': array([110573,   8699,   8678]), 'valid_labels': array([13978,  1071,  1021]), 'test_labels': array([13786,  1121,  1040]), 
    
    'train_valid_overlap': count    2390.000000
        mean        0.176342
        std         0.125046
        min         0.087121
        25%         0.119792
        50%         0.119792
        75%         0.159722
        max         0.958333
        dtype: float64, 
    
    'train_test_overlap': count    2390.000000
        mean        0.128636
        std         0.093485
        min         0.045455
        25%         0.078125
        50%         0.089286
        75%         0.141667
        max         0.458333
    """

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
