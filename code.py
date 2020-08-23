import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:\\Users\chinn\Desktop\dataset1.csv')
dataset.head()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
humidity=input("Enter Humidity value:")
soil=input("Enter soil type")
moisture=input("Enter moisture percentage")
temperature=input("Temperature")
rainfall=input("Enter rainfall level")
user=[humidity,soil,moisture,temperature,rainfall]
y_user = classifier.predict(user)
print(y_user)
error = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
dic={'wheat':'atrazine,furidone,metolachlor','rice':'malathion,carbaryl,lambda-cyhalothrin','cotton':'phorate,endosulfan,aldicarb','potato':'pyrethrin','tea':'pyrethrin','tomato':'copperspray','jasmine':'chrysanthenum flowertea,fenpropathran salt spray','jute':'endosulfan','sugarcane':'ametrya,msma','coffee':'olyphosate','rubber':'bitenthrin,chlorpyrifos,coumapho','tobacco':'nicotin','mangoes':'yates liquid copper','chillies':'avermuctins,chlortyritos,profenofos','corn':'aimec,anthen,cadet'}
for i in dic.keys():
    if(i==y_user):
        print("You can use following pesticides:",dic[i])
dic2={'wheat':'23.5','rice':'50','cotton':'5900 per quintal','groundnut':'55','potato':'10','sugarcane':'31','coffee':'230','mangoes':'50','coconut':'18 per piece','tea':'200','chillies':'90','corn':'30','tomato':'60','jasmine':'700','jute':'30','rubber':'150','tobacco':'151'}
for i in dic2.keys():
    if(i==y_user):
        print("market price:",dic2[i])
