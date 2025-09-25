
import numpy as np
import pickle
import statistics

# Define the file path structure
file_path_template = "Slicing_vectors_quantiles/{datasetname}{filtration}.pkl"


# Function to load dataset by name
def load_dataset_ricci_and_degcent(datasetname, m):
    # Construct the file path
    file_path = file_path_template.format(datasetname=datasetname, filtration='ricci')
    file_path1 = file_path_template.format(datasetname=datasetname, filtration='degcent')

    try:
        # Load the CSV file into a DataFrame
        with open(file_path, 'rb') as g:
            features_ricci = pickle.load(g)
        with open(file_path1, 'rb') as g:
            features_degcent = pickle.load(g)

        number_graph = len(features_ricci[m][0])
        number_slices = len(features_ricci[m][0][0])
        feat_ricci = [
            [[features_ricci[m][k][i][j] for k in range(4)] for j in range(number_slices)]
            for i in range(number_graph)
        ]
        feat_degcent = [
            [[features_degcent[m][k][i][j] for k in range(4)] for j in range(number_slices)]
            for i in range(number_graph)
        ]
        print(f"Successfully loaded dataset: {datasetname} features")
        return feat_ricci, feat_degcent
    except FileNotFoundError:
        print(f"Error: File for dataset '{datasetname}' not found.")
        return None


# Function to load dataset by name
def load_dataset_ricci_and_hks(datasetname, m):
    # Construct the file path
    file_path = file_path_template.format(datasetname=datasetname, filtration='ricci')
    file_path1 = file_path_template.format(datasetname=datasetname, filtration='hks')

    try:
        # Load the CSV file into a DataFrame
        with open(file_path, 'rb') as g:
            features_ricci = pickle.load(g)
        with open(file_path1, 'rb') as g:
            features_hks = pickle.load(g)

        number_graph = len(features_ricci[m][0])
        number_slices = len(features_ricci[m][0][0])
        number_slices_hks = len(features_hks[0][0])
        # print(number_slices )
        # print(number_slices_hks)
        feat_ricci = [
            [[features_ricci[m][k][i][j] for k in range(4)] for j in range(number_slices)]
            for i in range(number_graph)
        ]
        feat_hks = [
            [[features_hks[k][i][j] for k in range(4)] for j in range(number_slices)]
            for i in range(number_graph)
        ]
        # print(f"Successfully loaded dataset: {datasetname} features")
        return feat_ricci, feat_hks
    except FileNotFoundError:
        print(f"Error: File for dataset '{datasetname}' not found.")
        return None


# Function to load dataset by name
def load_dataset_degcent_and_hks(datasetname, m):
    # Construct the file path
    file_path = file_path_template.format(datasetname=datasetname, filtration='degcent')
    file_path1 = file_path_template.format(datasetname=datasetname, filtration='hks')

    try:
        # Load the CSV file into a DataFrame
        with open(file_path, 'rb') as g:
            features_degcent = pickle.load(g)
        with open(file_path1, 'rb') as g:
            features_hks = pickle.load(g)

        number_graph = len(features_degcent[m][0])
        number_slices = len(features_degcent[m][0][0])
        number_slices_hks = len(features_hks[0][0])
        # print(number_slices )
        # print(number_slices_hks)
        feat_degcent = [
            [[features_degcent[m][k][i][j] for k in range(4)] for j in range(number_slices)]
            for i in range(number_graph)
        ]
        feat_hks = [
            [[features_hks[k][i][j] for k in range(4)] for j in range(number_slices)]
            for i in range(number_graph)
        ]
        # print(f"Successfully loaded dataset: {datasetname} features")
        return feat_degcent, feat_hks
    except FileNotFoundError:
        print(f"Error: File for dataset '{datasetname}' not found.")
        return None


def load_label(dataset):
    if dataset=='PROTEINS':
        url='label/PROTEINS_graph_labels.txt'
        graph_label=np.loadtxt(url)
        max_value = np.max(graph_label)
        graph_label[graph_label == max_value] = 0 #start graph label with 0
    elif dataset=='BZR':
        url='label/BZR_graph_labels.txt'
        graph_label=np.loadtxt(url)
        min_value = np.min(graph_label)
        graph_label[graph_label == min_value] = 0 #start graph label with 0
    elif dataset=='COX2':
        url='label/COX2_graph_labels.txt'
        graph_label=np.loadtxt(url)
        min_value = np.min(graph_label)
        graph_label[graph_label == min_value] = 0 #start graph label with 0
    elif dataset=='MUTAG':
        url='label/MUTAG_graph_labels.txt'
        graph_label=np.loadtxt(url)
        min_value = np.min(graph_label)
        graph_label[graph_label == min_value] = 0 #start graph label with 0
    elif dataset=='IMDB-BINARY':
        url='label/IMDB-BINARY_graph_labels.txt'
        graph_label=np.loadtxt(url)
    elif dataset=='IMDB-MULTI':
        url='label/IMDB-MULTI_graph_labels.txt'
        graph_label=np.loadtxt(url)
        max_value = np.max(graph_label)
        graph_label[graph_label == max_value] = 0 #start graph label with 0
    elif dataset=='REDDIT-BINARY':
        graph_label=np.loadtxt('label/REDDIT-BINARY_graph_labels.txt')
        min_value = np.min(graph_label)
        graph_label[graph_label == min_value] = 0 #start graph label with 0
    elif dataset=='REDDIT-MULTI-5K':
        graph_label=np.loadtxt('label/REDDIT-MULTI-5K_graph_labels.txt')
        max_value = np.max(graph_label)
        graph_label[graph_label == max_value] = 0 #start graph label with 0
    else:
        print('Label not avilable')
#     graph_label=np.loadtxt(url)
#     max_value = np.max(graph_label)
#     graph_label[graph_label == max_value] = 0 #start graph label with 0
    return graph_label