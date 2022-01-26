import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from chemtonic.featurizer.RDKitMD.getRDKitMD import *
from chemtonic.curation.utils import molStructVerify
from .utils import suggest_K, visualizeElbow

#==========================================================
def suggest_num_clusters(compounds, 
                         estimated_num_cluster=10, 
                         thresold=0.05, 
                         visualize=True, 
                         exportImage=True, 
                         outputPath=None,
                         ignoreFailedStruct=False, 
                         getFailedStruct=False): 
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
                print("2. Get your errored SMILES(s) to check by setting 'getFailedStruct = True'")
                return None
            else:
                VerifiedCompounds = molStructVerify(compounds, printlogs=False)
    print("==============================================")
    #------------------------
    # Extract RDKitMD-features of input SMILES(s)
    print("Start extract RDKitMD features for input SMILES(s)")
    RDKitMD_features_df = extract_RDKitMD(VerifiedCompounds)
    RDKitMD_features_np = RDKitMD_features_df.iloc[:, 1:].to_numpy()
    # Normalize data
    scaler = StandardScaler()
    data_normal = scaler.fit_transform(RDKitMD_features_np) 
    #------------------------
    # Elbow method
    cost =[]
    for i in range(1, estimated_num_cluster):
        KM = KMeans(n_clusters = i, max_iter = 1000)
        KM.fit(data_normal)
        # Calculate squared errors for the clustered points
        cost.append(KM.inertia_)
    print("==============================================")
    #------------------------
    # Suggestion numbers of cluster (k) 
    k = suggest_K(list_values=cost, thresold=thresold)
    print("The suggested 'k' value (corresponding to elbow point): {}. \n Please see the Elbow cost plot for more details".format(k))
    print("==============================================")
    #------------------------
    # Visualize Elbow
    if visualize:
        visualizeElbow(cost, 
                       visualize, 
                       exportImage,
                       outputPath, 
                       estimated_num_cluster)
        return k
    else:
        return k
        
