import joblib
# import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv("heart.csv")

data.drop_duplicates(inplace=True)

cate_val = []
cont_val = []

for column in data.columns:
    if data[column].nunique() <= 10:
        cate_val.append(column)
    else:
        cont_val.append(column)

cate_val.remove("sex")
cate_val.remove("target")

data = pd.get_dummies(data=data, columns=cate_val, drop_first=True)

st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log = LogisticRegression()
log.fit(X_train, y_train)
y_predict = log.predict(X_test)

svm = SVC()
svm.fit(X_train, y_train)
y_predict2 = svm.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_predict3 = knn.predict(X_test)


# non linear algorithme
data = pd.read_csv("heart.csv")
data.drop_duplicates(inplace=True)
X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_predict4 = dt.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_predict5 = rf.predict(X_test)


gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_predict6 = gbc.predict(X_test)


final_data = pd.DataFrame(
    data={
        "Models": ["LR", "SVM", "KNN", "DT", "RF", "GB"],
        "ACC": [
            accuracy_score(y_true=y_test, y_pred=y_predict),
            accuracy_score(y_true=y_test, y_pred=y_predict2),
            accuracy_score(y_true=y_test, y_pred=y_predict3),
            accuracy_score(y_true=y_test, y_pred=y_predict4),
            accuracy_score(y_true=y_test, y_pred=y_predict5),
            accuracy_score(y_true=y_test, y_pred=y_predict6)
        ]
    }
)

sns.barplot(data=final_data, x="Models", y="ACC")
plt.show()

X = data.drop("target", axis=1)
y = data["target"]

rf = RandomForestClassifier()
rf.fit(X, y)

new_data = pd.DataFrame(
    data={
     'age': 52,
     'sex': 1,
     'cp': 0,
     'trestbps': 125,
     'chol': 212,
     'fbs': 0,
     'restecg': 1,
     'thalach': 168,
     'exang': 0,
     'oldpeak': 1,
     'slope': 2,
     'ca': 2,
     'thal': 3,
    },
    index=[0]
)

p = rf.predict(new_data)
if p[0] == 0:
    print("No Disease")
else:
    print("Disease")

joblib.dump(rf, "model_")
