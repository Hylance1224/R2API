# R2API: A novel method for web API recommendation by using HGNNs with multi-task learning

Here is the relevant dataset and open-source code for the article titled "R2API: A novel method for web API recommendation by using HGNNs with multi-task learning"

## 1. R2API

### 1.1 Usage

1.  **Model training**

    Simply start Python in COMMAND LINE mode, then use the following statement (one line in the COMMAND Prompt window) to execute **main.py**:

        python main.py --dataset fold_1 --weight_decay 0.0001 --lr 0.01 
        --pretrain 0 --save_flag 1

2.  **Model testing**

    Start Python in COMMAND LINE mode, then use the following statement (one line in the COMMAND Prompt window) to execute **test\_model.py**:

        python generate_recommendation.py --dataset fold_1 --output_path output

    Once the program execution is complete, it will generate a folder named "output", where the recommendation results is stored.

### 1.2 Environment Settup

Our code has been tested under Python 3.12.3. The experiment was conducted via PyTorch, and thus the following packages are required:

    torch == 1.3.1
    numpy == 1.18.1
    scipy == 1.3.2
    sklearn == 0.21.3

Updated version of each package is acceptable.&#x20;

### 1.3 Description of Essential Folders and Files

| Name                        | Type   | Description                                                                                                                                                                                                                                                                                                                    |
| --------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| data                        | Folder | Data files required for the experiment. Specifically: **api\_tag\_vector.h5** stores the semantic vectors of the API tags, **vector.h5** stores the semantic vectors of mashup descriptions, **api\_vector.h5** stores the semantic vectors of API descriptions, **mashup\_tag.h5** stores the semantic vectors of mashup tags |
| original data               | Folder | Save the data related to mashups and APIs used in the experiment, including the invocation relationships between mashups and APIs, the descriptions and tags of mashups, and the descriptions and tags of APIs.                                                                                                                |
| main.py                     | File   | Model training python file of AttenTPL                                                                                                                                                                                                                                                                                         |
| generate\_recommendation.py | File   | Python file used for generating the web API recommendation                                                                                                                                                                                                                                                                     |
| Models.py                   | File   | Model modules of R2API                                                                                                                                                                                                                                                                                                         |
| utility                     | Folder | Tools and essential libraries used by AttenTPL                                                                                                                                                                                                                                                                                 |

