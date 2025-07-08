# Graph-Based Node Classification with GCN, GAT, and GraphSAGE on the Cora Dataset
This project implements and evaluates three popular Graph Neural Network (GNN) architectures for the task of node classification on the Cora citation dataset. The models explored are the Graph Convolutional Network (GCN), the Graph Attention Network (GAT), and GraphSAGE.
## The Cora Dataset
The Cora dataset is a widely used benchmark for graph-based machine learning tasks. It consists of a network of scientific publications and their citation relationships.

- Nodes: 2,708 scientific publications.

- Edges: 5,429 citation links between publications.

- Node Features: Each publication is represented by a 1,433-dimensional binary vector, indicating the presence or absence of words from a predefined dictionary.

- Classes: Each paper is classified into one of seven distinct research areas

The primary task on this dataset is to predict the research area of a given publication based on its content (node features) and its connections within the citation network (graph structure).

## Model Architectures
### 1. Graph Convolutional Network (GCN)
The GCN is a foundational GNN model that learns node representations by aggregating information from its local neighborhood. The architecture used in this project is a simple yet effective two-layer GCN.

``` GCNConv(dataset.num_node_features, hidden_channels)``` :
 The first graph convolutional layer takes the node features as input and transforms them into a lower-dimensional representation (hidden_channels).

ReLU(): A Rectified Linear Unit activation function is applied element-wise to introduce non-linearity.

Dropout(): A dropout layer is applied to the hidden representations to prevent overfitting by randomly setting a fraction of neuron activations to zero during training.

``` GCNConv(hidden_channels, dataset.num_classes)``` 
: The second graph convolutional layer takes the hidden representations and maps them to the final number of output classes.

A hyperparameter search was conducted for the GCN model to find the optimal combination of hidden units, learning rate, and dropout rate. The best performing configuration was:

Hidden Units: 64

Learning Rate: 0.01

Dropout: 0.5

Best Validation Accuracy: 80.2%

### 2. Graph Attention Network (GAT)
The GAT model enhances the GCN by incorporating an attention mechanism. This allows the model to assign different levels of importance to different nodes in a neighborhood when aggregating information.

``` GATConv(dataset.num_node_features, hidden_channels, heads=heads, dropout=0.3)```
: The first GAT layer computes attention scores between a central node and its neighbors. It utilizes multi-head attention (heads) to capture different types of relationships. Dropout is also applied within the attention mechanism.

```python ELU()```: The Exponential Linear Unit (ELU) activation function is used, which is common in GAT architectures.

``` GATConv(hidden_channels * heads, dataset.num_classes, heads=1, concat=False, dropout=0.3)``` : The second GAT layer aggregates the information from the different attention heads and produces the final class predictions.

The GAT model achieved a test accuracy of 79.2%.

### 3. GraphSAGE
GraphSAGE (Sample and Aggregate) is another popular GNN architecture that is particularly effective for large graphs. It learns a function to generate node embeddings by sampling and aggregating features from a node's local neighborhood.

``` SAGEConv(dataset.num_node_features, hidden_channels, aggr='mean')``` : The first GraphSAGE convolutional layer aggregates features from the neighbors using a mean aggregator.

ReLU(): A ReLU activation function is applied.

Dropout(): A dropout layer is used for regularization.

``` SAGEConv(hidden_channels, dataset.num_classes, aggr='mean')```: The second GraphSAGE layer performs another round of aggregation and produces the final class logits.

The GraphSAGE model demonstrated strong performance with a test accuracy of 80.3%.
