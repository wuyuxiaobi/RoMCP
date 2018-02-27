# RoMCP
RoMCP is a new representation about medical features. It can capture both diagnosis information and temporal relations between days. The learned diagnosis embedding grasps the key factors of the disease, and each day embedding is determined by the diagnosis together with the preorder days.

---

### Running RoMCP

#### Step1: Installation
&emsp;&emsp;1.Install python, Tensorflow. We use Python 2.7, Tensorflow 1.3.0.

&emsp;&emsp;2.If you want to use GPU to accelerate computation, install CUDA

&emsp;&emsp;3.Download/Clone the RoMCP code

#### Step2: Preparing data 

To run RoMCP, three kinds of data need to be prepared. 

&emsp;&emsp;1. D2bow(Day to bow): The first data file names "d2bow" which record medical activity for everyday of all visits. D2bow is a two dimension list, spliting every visits by -1. About medical acitvity for a day, it consists of a series of tuples. The tuple has two values, one is medical acitvity index(code) and the other is dosage after normalization.

&emsp;&emsp;For example:

    [
        [(2,0.8236),(6,0.7632),(100,0.6666),(4212,0.1566)],
        [(9, 0.5), (12, 0.7048), (14, 0.5)]
        [(6, 0.5), (7, 0.7048), (9, 0.5), (12, 0.7048), (14, 0.5)],
        -1,
        [(5, 0.7048), (12, 0.5), (14, 0.5), (18, 0.5)],
        [(2, 0.9745), (3, 0.5), (36, 0.5), (37, 0.5)]
    ]
&emsp;&emsp; Above is a example of d2bow, which represents 2 visits, one has 3 days and the other has 2 days.

&emsp;&emsp;2. D2diag(Day to diagnosis): Second data file names "d2diag", recording diagnosis index(code) of everyday. D2diag is a one dimension list, spliting every visits by 0. In a visit, everyday has some diagnosis code.

&emsp;&emsp; For example:

    [188,188,188,0,2,2]
&emsp;&emsp; Above example represents 2 visits, the first visit has three days and the diagnosis code is 188, the second visit has two days and the diagnosis code is 2.

&emsp;&emsp;3. Mask: Thrid data file names "mask". It is a indicator to use aggreate data by win_size. The detail about how to use the mask is coded in the "Disease2Vec.py" function _init_aggregate_day(). Mask is a two dimension list, spliting every visits by [0]. In mask, a day maps to [1].

&emsp;&emsp; For example:

    [[1],[1],[1],[0],[1],[1]]
&emsp;&emsp; Above example represents 2 visits, the first visit has three days and the second visit has two days .

#### Step3: Training model
After prepared the data, you can feed the data into the function main() in "Disease2VecRunner.py". In function main(), it splits your data into train set and test set and show the NDCG and Recall score about the next day medical activity prediction.
