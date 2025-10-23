
# TOPOLOGY, MEET ATTENTION: THE TOPOFORMER FOR GRAPH LEARNING
Welcome to the **TOPOFORMER** repository! Topo-TR is an innovative machine learning framework designed for graph property prediction. It seamlessly integrates *topological data analysis (TDA)* with *transformer* architectures to enhance graph learning.

At the core of Topo-TR is **Topo-Scan**, a novel technique that efficiently encodes rich topological signatures into a sequential format, boosting model performance.

This implementation includes:

- 9 benchmark datasets for graph property prediction
- 7 datasets for molecular property prediction

The code is written in Python and leverages PyTorch and PyTorch Geometric for efficient computation.

![TopoTR](https://github.com/user-attachments/assets/a7e5e309-9cde-4a56-8aa5-beb06b0faba8)



# Model Architecture
- **Topo-Scan Encoding & Topological Feature Extraction**

   - A given graph is processed using different filtrations to extract topological signatures with *Topo-scan*.
   - Each filtration generates a topological sequence, capturing rich structural features of the graph.

   - The extracted topological sequences are converted into embeddings, making them suitable for transformer processing.

- **Transformer-Based Feature Processing**
  - Each topological sequence is passed through a transformer encoder, consisting of:
    - Embedding & Projection Layers
    - Multi-Headed Self-Attention Mechanisms
    - Feedforward Networks
    - Layer Normalization & Residual Connections
  - The transformer architecture processes different topological filtrations separately.
- **Attention-Based Feature Fusion**
  
The output representations from different transformers are concatenated using an attention-based mechanism, ensuring optimal feature integration.
- **Graph Property Prediction**

The fused topological representation is passed through a final dense layer to predict the target graph property.


# Requirements
Wise-GNN depends on the followings:
Pytorch 2.4.0, Pyflagser 0.4.7, networkx 3.2.1, sklearn 1.3.0, torch_geometric 2.4.0

   
The code is implemented in python 3.11.4. 

# Runing the  Experiments
To repeat the experiment a specific dataset, run the train_*.py file
  

# Contributing
Contributions are welcome! Please follow these steps:

- Fork the repository.
- Create a new branch for your feature or bugfix.
- Submit a pull request with detailed changes.
- Feel free to open issues for discussion or questions about the code.

