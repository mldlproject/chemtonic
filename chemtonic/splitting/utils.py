import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import random

from chemtonic.featurizer.RDKitMD.getRDKitMD import *
from chemtonic.curation.utils import molStructVerify

#==========================================================
# Get samples from each cluster to create train, val, test sets
def sampling_from_clusters(list_index, 
                           n_clusters, 
                           data_size, 
                           sample_size, 
                           pre_index=None, 
                           seed=0):
    #---------------------------------------
    random.seed(seed) 
    count = 0
    index_sample = []
   #---------------------------------------
    for k in range(n_clusters):
        # Calculate cluster size based on input sample size
        #---------------------------------------
        cluster_size = int(len(list_index[k])/data_size * sample_size)
        if k == n_clusters - 1:
            cluster_size = sample_size - count
        #---------------------------------------
        i = 0
        while i < cluster_size:
            # Random cluster starting at 0
            smiles_idx = random.choice(list_index[k]) #consider random.sample()
            if smiles_idx not in pre_index and smiles_idx not in index_sample:
                index_sample.append(smiles_idx)
                i +=1
        count += cluster_size
        #---------------------------------------
    return index_sample

#==========================================================
# Suggest number of clusters
def suggest_K(list_values, thresold=0.05):
    n_cluster = 0 
    for idx in range(len(list_values)-2):
        # Examine midle point 
        decide_idx = idx + 1
        next_idx = idx+2

        # Calculate the decrement at each point (in consideration)
        ratio_error_1 = 1 - list_values[decide_idx]/list_values[idx]
        ratio_error_2 = 1 - list_values[next_idx]/list_values[decide_idx]
        if ratio_error_1 <= 0 or ratio_error_2 <= 0:
            continue

        if ratio_error_2 <= thresold:
            n_cluster = decide_idx + 1
            return n_cluster
    # Can't find suitable results
    print("The selected threshold requires the estimated number of clusters (k) to be enlarged. \n Please add another number of clusters or increase the threshold")
    return

#==========================================================
# PCA_method 
def visualizePCA(compounds, 
                 num_clusters, 
                 exportImage=False,
                 outputPath=None,
                 ignoreFailedStruct=False, 
                 getFailedStruct=False,
                 figsize=(6,6),
                 color_palette = "viridis"): 
    #------------------------
    if exportImage:
        if outputPath == None:
            print("!!!ERROR 'exportImage=True' needs 'outputPath=<Directory>' to be filled !!!")
            return None 
    if outputPath:
        if exportImage == False:
            print("!!!ERROR 'outputPath=<Directory>' needs to set 'exportImage=True' !!!")
            return None 
    #------------------------
    # Conver input to list 
    if isinstance(compounds, pd.core.series.Series):
        compounds = compounds.tolist()
    if isinstance(compounds, pd.core.frame.DataFrame):
        compounds = compounds.iloc[:,0].tolist()
    if isinstance(compounds, str):
        compounds = [compounds]
    if isinstance(compounds, list):
        compounds = compounds
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
    # Extract RDKitMD-features of input SMILES(s)
    print("Start extracting RDKitMD features for input SMILES(s)")
    RDKitMD_features_df = extract_RDKitMD(VerifiedCompounds)
    RDKitMD_features_np = RDKitMD_features_df.iloc[:, 1:].to_numpy()
    # Normalizing data
    scaler = StandardScaler()
    data_normal = scaler.fit_transform(RDKitMD_features_np) 
    # PCA
    pca = PCA(n_components= 2)
    principalComponents = pca.fit_transform(data_normal)
    PCA_components = pd.DataFrame(principalComponents)
    #------------------------
    try:
        model = KMeans(n_clusters= num_clusters)
        model.fit(PCA_components.iloc[:,:2])
    except:
        print("The number of clusters does not match, please choose the another value")
        return None
    labels = model.predict(PCA_components.iloc[:,:2])
    labels_cluster = ['Cluster {}'.format(i) for i in range(1, num_clusters+1)]
    plt.figure(figsize=figsize)
    plot1 = plt.scatter(PCA_components[0], PCA_components[1], c=labels, cmap = sns.color_palette(color_palette, as_cmap=True))
    plt.legend(handles=plot1.legend_elements()[0], labels=labels_cluster)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    #------------------------
    if exportImage:
        filePath = outputPath + "plot_PCA_cluster.pdf"
        if os.path.isdir(outputPath):
            plt.savefig(filePath)
            plt.show()
        else:
            os.makedirs(outputPath)
            plt.savefig(filePath)
            plt.show()
    else:
        plt.show()
    
#==========================================================
# Eblow method 
def visualizeElbow(cost, 
                   visualize=True,
                   exportImage=True, 
                   outputPath=None,
                   estimated_num_cluster=10):
    #------------------------
    if exportImage:
        if outputPath == None:
            print("!!!ERROR 'exportImage=True' needs 'outputPath=<Directory>' to be filled !!!")
            return None 
    if outputPath:
        if exportImage == False:
            print("!!!ERROR 'outputPath=<Directory>' needs to set 'exportImage=True' !!!")
            return None 
    #------------------------
    plt.figure(figsize=(7,6))
    # plot the cost against K values
    x = np.arange(1,estimated_num_cluster)
    plt.plot(x, cost, color='g', linewidth=3)
    plt.xticks(np.arange(1,estimated_num_cluster), x)
    plt.xlabel("Value of $\mathit{k}$")
    plt.ylabel("Squared Error (Cost)")
    plt.legend()
    #------------------------
    if exportImage:
        filePath = outputPath + "plot_squared_error.pdf"
        if os.path.isdir(outputPath):
            plt.savefig(filePath)
            if visualize:
                plt.show()
        else:
            os.makedirs(outputPath)
            plt.savefig(filePath)
            if visualize: 
                plt.show()
    else:
        if visualize:
            plt.show()
