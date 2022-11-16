### Data Wrangling
import pandas as pd
import numpy as np
from collections import OrderedDict

### Modelling
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

### Remove unnecessary warnings
import warnings

warnings.filterwarnings('ignore')

# load data set
dataset_url = "https://github.com/MuhammadFaisalAvicenna/datamining/blob/master/CreditScore_test.csv"

df = pd.read_csv(dataset_url)
df.head()

df.shape
