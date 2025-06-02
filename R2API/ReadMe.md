# R2API: A novel method for web API recommendation by using HGNNs with multi-task learning

Here is the relevant dataset and open-source code for the article titled "R2API: A novel method for web API recommendation by using HGNNs with multi-task learning"

## 1. R2API

### 1.1 Environment Settup

Our code has been tested under **Python 3.12.3**, and the following packages are required:

    h5py==3.11.0
    scikit_learn==1.4.2
    sentence_transformers==2.7.0
    torch==2.7.0+cu126

### 1.2 Usage

1.  **Package installation**

        pip install -r requirements.txt

2.  **Data Generation (Optional)**&#x20;

    > This step can be skipped, as all necessary training and testing files have already been prepared for the experiments in this paper. However, if you wish to run the experiments on a new dataset, you can execute the following script.

    To generate the dataset, start Python in **command line** mode and run:

        python generate_dataset.py

    This script uses the original files `Original Dataset/shuffle_mashup.json` and `Original Dataset/api.json` as input and performs the following operations:

    *   Generates **10-fold cross-validation data** under the `training_dataset` folder.

    *   Creates a `data` folder containing:

        *   `API_vectors.h5` and `vectors.h5`: Sentence embedding vectors for API and mashup descriptions.

        *   `api_tag_vector.h5` and `mashup_tag_vector.h5`: Embedding vectors for API and mashup tags.

    Each fold directory (`fold_1` to `fold_10`) includes the following files:

    *   `RS.csv`: Test set of mashup–API pairs.

    *   `TE.csv`: Training set of mashup–API pairs.

    *   `api_tags.csv`, `Api_tag_mapping.csv`: API–tag relationships and tag–index mappings.

    *   `mashup_tags.csv`, `mashup_tag_mapping.csv`: Mashup–tag relationships and tag–index mappings.

3.  **Model Training and Testing**

    To train and test the model on a specific fold, start Python in **command line** mode and execute the following (in one line):

        python main.py --dataset fold_1

    Here, `fold_1` can be replaced with `fold_2`, `fold_3`, ..., up to `fold_10`, corresponding to the ten folds of the dataset.

    Upon completion, the results will be saved in a folder named `output`, which contains the recommendation results for the specified fold.

4.  **Evaluation Metrics**

    Once training and testing are complete for a given fold, you can **evaluate the model’s performance** by calculating the corresponding metrics. Use the command below:

        python metrics.py --dataset fold_1

    **Make sure that the specified fold (**`fold_1`**, **`fold_2`, etc.) matches the one used in the training and testing step.

    You will obtain output in the following format:

        The performance of fold_1 is as follows: 
        N=3 -> Precision: 0.3111, Recall: 0.6639, MAP: 0.7231, NDCG: 0.6675, Cov: 0.1798 
        N=5 -> Precision: 0.2022, Recall: 0.7088, MAP: 0.7249, NDCG: 0.6823, Cov: 0.2585 
        N=10 -> Precision: 0.1116, Recall: 0.7593, MAP: 0.7157, NDCG: 0.7007, Cov: 0.4032 
        N=20 -> Precision: 0.0598, Recall: 0.7975, MAP: 0.7071, NDCG: 0.7123, Cov: 0.5883

    These results reflect the performance of the model on **fold\_1**.

### 1.3 Description of Essential Folders and Files

| Name          | Type   | Description                                                                                                                                                                                                                                                                                                                    |
| ------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| data          | Folder | Data files required for the experiment. Specifically: **api\_tag\_vector.h5** stores the semantic vectors of the API tags, **vector.h5** stores the semantic vectors of mashup descriptions, **api\_vector.h5** stores the semantic vectors of API descriptions, **mashup\_tag.h5** stores the semantic vectors of mashup tags |
| original data | Folder | Save the data related to mashups and APIs used in the experiment, including the invocation relationships between mashups and APIs, the descriptions and tags of mashups, and the descriptions and tags of APIs.                                                                                                                |
| main.py       | File   | Model training and testing python file of R2API                                                                                                                                                                                                                                                                                |
| Models.py     | File   | Model modules of R2API                                                                                                                                                                                                                                                                                                         |
| utility       | Folder | Tools and essential libraries used by R2API                                                                                                                                                                                                                                                                                    |

####

