This is the repository of the TSC paper `Service Recommendations for Mashup Based on Generation Model`.

## dataset

Dataset refers to `Mashup-Oriented Web API Recommendation via Multi-Model Fusion and Multi-Task Learning` and `Revisiting, Benchmarking and Exploring API Recommendation: How Far Are We?`

***

## **🔧 Supplementary Instructions (added by Yihui Wang**👩‍💻)

The following section provides additional guidance for setting up and running the project.

### 1. Python Version Requirements and Dependency Installation

&#x20;

**Python 3.8.0 environment**\
Required to run data generation and semantic-related scripts.\
Install dependencies with the following command (please adjust the Python path according to your local installation):

    D:\Python\Python38\python.exe -m pip install -r requirements_python38.txt

**Python 3.7.0 environment**\
Used for training the recommendation model.\
Install dependencies with the following command (please adjust the Python path according to your local installation):

    D:\Python\Python37\python.exe -m pip install -r requirements_python37.txt

### 2. Running Steps

Please use the matching Python version to run the corresponding scripts. Examples (modify the paths according to your Python installation):

1.  **Data Generation (Python 3.8):**

<!---->

    D:\Python\Python38\python.exe generate_data.py

1.  **file downloading and unzip**

**Download** the file from:\
<http://nlp.stanford.edu/data/glove.6B.zip>

**Unzip** it and place the file `glove.6B.200d.txt` into the folder named `.vec_cache` under the project root.



1.  **Semantic Processing Scripts (Python 3.8):**

<!---->

    D:\Python\Python38\python.exe retrieve/bert_whitening.py
    D:\Python\Python38\python.exe retrieve/Semantic.py

1.  **Train Recommendation Model (Python 3.7):**

<!---->

    D:\Python\Python37\python.exe train_recommder_api.py

### 3. Notes

**Different Python Installation Paths**\
If your Python installation path differs from the examples above, please replace the paths `D:\Python\Python38\python.exe` and `D:\Python\Python37\python.exe` with your actual Python interpreter paths.\
For example:

On Windows, Python is usually located at\
`C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python38\python.exe`

Or, if added to the system environment variables, you can directly use the commands `python3.8` or `python3.7`.

**Virtual Environment Recommended**\
It is recommended to create separate virtual environments for Python 3.7 and Python 3.8 to isolate dependencies and avoid version conflicts.\
You can use `venv` or `conda` to create virtual environments.
-------------------------------------------------------------

