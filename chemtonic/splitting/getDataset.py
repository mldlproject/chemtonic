import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from chemtonic.featurizer.RDKitMD.getRDKitMD import *
from chemtonic.curation.utils import molStructVerify
from .utils import sampling_from_clusters

#==========================================================
# Define clusters using k-mean clustering algorithm
def split_data(compounds, 
               num_clusters=3, 
               num_datasets=2, 
               ratio_list = [0.5, 0.5], 
               ignoreFailedStruct=False, 
               getFailedStruct=False, 
               exportCSV=False, 
               outputPath=None, 
               seed=0):
    #------------------------
    if exportCSV:
        if outputPath == None:
            print("!!!ERROR 'exportCSV=True' needs 'outputPath=<Directory>' to be filled !!!")
            return None 
    if outputPath:
        if exportCSV == False:
            print("!!!ERROR 'outputPath=<Directory>' needs to set 'exportCSV=True' !!!")
            return None 
    #------------------------
    # Check input 
    if num_datasets < 2 or num_datasets >5:
        print("Please select 'num_clusters' from 2 to 5")
        return None
    if len(ratio_list) != num_datasets:
        print("The number of to-be-created datasets must be equal to the number of ratios. \n E.g., If selecting 'num_datasets=2', please selecting 'ratio_list = [0.5, 0.5]'.")
        return None
    try:
        total_ratio = sum(ratio_list)
        if total_ratio != 1.0:
            print("The sum of all ratios must be equal to 1")
            return None
    except:
        print("Please define 'ratio_list = [r1, r2, ..., r5]' so that (r1 + r2 + ... + r5) = 1")
        return
    #------------------------
    # Convert input to list 
    if isinstance(compounds, pd.core.series.Series):
        compounds = compounds.tolist()
    if isinstance(compounds, pd.core.frame.DataFrame):
        compounds = compounds.iloc[:,0].tolist()
    if isinstance(compounds, str):
        compounds = [compounds]
    if isinstance(compounds, list):
        compounds = compounds
    #------------------------
    # Check valid compounds  
    Unverified_count  = len(molStructVerify(compounds, getFailedStruct=True, printlogs=False))
    if Unverified_count == len(compounds):
        print("There are no valid SMILES found, please check!")
        return None
    if ignoreFailedStruct:
        VerifiedCompounds = molStructVerify(compounds, printlogs=False)
        if Unverified_count != 0:
            print("There are {} errored SMILES(s) which were/was ignored".format(Unverified_count))
    else:
        if getFailedStruct:
            if Unverified_count !=0:
                print("There are {} errored SMILES(s), to ignore them and continue running, please set 'ignoreFailedStruct=True'".format(Unverified_count))
                Unverified = molStructVerify(compounds, getFailedStruct=True, printlogs=False)
                return Unverified
            else:
                print("No errored SMILES found")
                VerifiedCompounds = molStructVerify(compounds, printlogs=False)
        else:
            if Unverified_count!=0:
                print("Your set of compounds contains errored SMILES(s), you can:")
                print("1. Ignore the errored SMILES(s) and continue running by setting 'ignoreFailedStruct=True'")
                print("2. Get your errored SMILES(s) to check by setting 'getFailedStruct=True'")
                return None
            else:
                VerifiedCompounds = molStructVerify(compounds, printlogs=False)
    print("==============================================")
    #------------------------
    # Extract RDKitMD-features of input SMILES(s)
    print("Start extract RDKitMD features for input SMILES(s)")
    RDKitMD_features_df = extract_RDKitMD(VerifiedCompounds)
    RDKitMD_features_np = RDKitMD_features_df.iloc[:, 1:].to_numpy()
    # Normalizing data
    scaler = StandardScaler()
    data_normal = scaler.fit_transform(RDKitMD_features_np) 
    # k-mean clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data_normal)
    list_index = []
    for i in range(num_clusters): 
        list_index.append(np.where(kmeans.labels_ == i)[0])
    #------------------------
    print("==============================================")
    print("Start splitting data")
    allDatasetsIdx = []
    allDatasets_df = []
    for i in range(num_datasets-1):
        num_sample = len(kmeans.labels_) * ratio_list[i]
        # Split data
        datasetIdx  = sampling_from_clusters(list_index, num_clusters, len(kmeans.labels_), num_sample, allDatasetsIdx, seed=seed)
        dataset_df = pd.DataFrame(RDKitMD_features_df, index=datasetIdx, columns=['SMILES']).reset_index(drop=True)
        allDatasets_df.append(dataset_df)
        allDatasetsIdx += datasetIdx

    finalDataset = []
    for i in range(len(data_normal)):
        # if  i not in list_index_test and i not in list_index_val:
        if i not in allDatasetsIdx:
            finalDataset.append(i)
    dataset_df = pd.DataFrame(RDKitMD_features_df, index=datasetIdx, columns=['SMILES']).reset_index(drop=True)
    allDatasets_df.append(dataset_df)
    print("----------------------------------------------")
    print("Successfully done!")
    #------------------------
    if exportCSV:
        for i in range(num_datasets):
            filePath = outputPath + "data_fold_{}_seed_{}.csv".format(i+1, seed)
            if os.path.isdir(outputPath):
                allDatasets_df[i].to_csv(filePath, index=False)
            else:
                os.makedirs(outputPath)
                allDatasets_df[i].to_csv(filePath, index=False)
        return allDatasets_df
    else:
        return allDatasets_df
