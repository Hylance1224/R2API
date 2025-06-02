# FSFM

This code is an implementation of the paper `Functional and Structural Fusion-based Web API Recommendations in Heterogeneous Networks`.\
It is adapted from the code of the paper `SEHGN: Semantic-Enhanced Heterogeneous Graph Network for Web API Recommendation`

## Usage

### 1. Python Version Requirements and Dependency Installation

**Recommended Environment:** Python 3.7.0\
To install the required dependencies, run the following command:

    pip install -r requirements.txt

### 2. Data generation&#x20;

To generate the necessary data, execute:

    python.exe generate_data.py

### 3. file downloading and unzip

**Download** the file from:\
<http://nlp.stanford.edu/data/glove.6B.zip>

&#x20;**Unzip** it and place the file `glove.6B.200d.txt` into the folder named `.vec_cache` under the project root.

### 4. Model training and testing

    python model/SEHGN.py

