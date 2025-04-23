# Ex-4
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
       import pandas as pd
       import numpy as np

       df=pd.read_csv("bmi.csv")
       df
![image](https://github.com/user-attachments/assets/e0ac108e-ce2b-44fd-a6ea-7554aec2a989)

       df.head()
![image](https://github.com/user-attachments/assets/bd273845-aedb-4a8f-ad3f-84be0792bf94)

      df.dropna()
![image](https://github.com/user-attachments/assets/0d31a7a4-ae00-40e2-8876-d65237cb3934)

       max_vals=np.max(np.abs(df[['Height','Weight']]))
       max_vals
![image](https://github.com/user-attachments/assets/9883ac1c-89d2-449d-976a-559327d40fbc)

       from sklearn.preprocessing import MinMaxScaler
       scaler=MinMaxScaler()
       df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])fr
       df.head(10)
![image](https://github.com/user-attachments/assets/3fd59ecd-003c-48fd-9c48-e07e0dd54547)

       df1=pd.read_csv("C:\\Users\\nikhil\\OneDrive\\Documents\\bmi (1).csv")
       df2=pd.read_csv("C:\\Users\\nikhil\\OneDrive\\Documents\\bmi (1).csv")
       df3=pd.read_csv("C:\\Users\\nikhil\\OneDrive\\Documents\\bmi (1).csv")
       df4=pd.read_csv("C:\\Users\\nikhil\\OneDrive\\Documents\\bmi (1).csv")
       df5=pd.read_csv("C:\\Users\\nikhil\\OneDrive\\Documents\\bmi (1).csv")
       df1

![388504056-a7322a98-3297-41b6-a5e8-595c91dc8b4d](https://github.com/user-attachments/assets/c54446bf-5c28-4e9e-8226-15399d71230c)

       from sklearn.preprocessing import StandardScaler
       sc=StandardScaler()
       df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
       df1.head(10)

![388506622-6ce0dc2e-72ee-4283-ab1a-61e105b7b596](https://github.com/user-attachments/assets/a8fa6e0d-47d8-4dd4-812f-9b6b49439dc0)

       from sklearn.preprocessing import Normalizer
       scaler=Normalizer()
       df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
       df2

![388506982-d5705138-76ec-4a7f-ab0a-04f1e473fdc6](https://github.com/user-attachments/assets/59a8a7c7-94e2-4733-a4e1-2d084d438bfc)

       from sklearn.preprocessing import MaxAbsScaler
       scaler=MaxAbsScaler()
       df3[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
       df3 

![388507343-074d8e0c-80fa-47c0-bac6-7353f1b39ab7](https://github.com/user-attachments/assets/e0064e1e-50e1-43d2-bbd7-3ddef3c0c8b7)

       from sklearn.preprocessing import RobustScaler
       scaler=RobustScaler()
       df4[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
       df4

![388507670-de4deeba-b8ab-4c5d-8818-6768a02c8543](https://github.com/user-attachments/assets/29000f9e-a97b-4af9-bf4e-b7a0a0f039d6)

       import seaborn as sns
       feature selection 
       import pandas as pd

       import numpy as np 
       import seaborn as sns
       import seaborn as sns
       import pandas as pd
       from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
       from sklearn.feature_selection import chi2
       data=pd.read_csv("C:\\Users\\nikhil\\Downloads\\titanic_dataset (1).csv")
       data

![388511413-c0330f51-a459-4e63-ba8f-8356dc929c69](https://github.com/user-attachments/assets/68e38afc-908e-4351-9682-7ba9ac40abde)

       data=data.dropna()
       x=data.drop(['Survived','Name','Ticket'],axis=1)
       y=data['Survived']
       data["Sex"]=data["Sex"].astype("category")
       data["Cabin"]=data["Cabin"].astype("category")
       data["Embarked"]=data["Embarked"].astype("category")
       data["Sex"]=data["Sex"].cat.codes
       data["Cabin"]=data["Cabin"].cat.codes
       data["Embarked"]=data["Embarked"].cat.codes
       data

![388515515-2a598b08-3121-478e-a22b-c7c0986f79c3](https://github.com/user-attachments/assets/804597ec-4494-4887-a695-ec5246e34daa)

       k=5 selector=SelectKBest(score_func=chi2, k=k) x=pd.get_dummies(x) x_new=selector.fit_transform(x,y)

       x_encoded =pd.get_dummies(x) selector=SelectKBest(score_func=chi2, k=5) x_new = selector.fit_transform(x_encoded,y)

       selected_feature_indices=selector.get_support(indices=True)
       selected_features=x.columns[selected_feature_indices]
       print("Selected_Feature:")
       print(selected_features)

![388516439-317c121c-abf4-4dc7-a14d-0f3e899e50b4](https://github.com/user-attachments/assets/595e1b6f-4a90-4620-92da-79bcc0570555)

       selector=SelectKBest(score_func=mutual_info_classif, k=5)
       x_new = selector.fit_transform(x,y)
       selected_feature_indices=selector.get_support(indices=True)
       selected_features=x.columns[selected_feature_indices]
       print("Selected Features:")
       print(selected_features)

![388516903-70e1b6b3-e6a9-4b0c-abf6-e257e8dff14f](https://github.com/user-attachments/assets/6d29ff33-87cd-45ed-a9a4-dbdfa4d86a19)

       selector=SelectKBest(score_func=mutual_info_classif, k=5)
       x_new = selector.fit_transform(x,y)
       selected_feature_indices=selector.get_support(indices=True)
       selected_features=x.columns[selected_feature_indices]
       print("Selected Features:")
       print(selected_features)

![388521115-1814553c-c56f-42a4-9416-589884be81bd](https://github.com/user-attachments/assets/071ae054-4fff-4224-93ce-9d76b83ccd21)

       from sklearn.feature_selection import SelectFromModel
       from sklearn.ensemble import RandomForestClassifier
       model=RandomForestClassifier()
       sfm=SelectFromModel(model,threshold='mean')
       x=pd.get_dummies(x)
       sfm.fit(x,y)
       selected_features=x.columns[sfm.get_support()]
       print("Selected Features:")
       print(selected_features)

![388517155-412dcc3f-3af8-4678-9da0-1a354173f174](https://github.com/user-attachments/assets/da4dd698-96c2-4104-86dd-f3e73534382e)

       from sklearn.ensemble import RandomForestClassifier
       model=RandomForestClassifier(n_estimators=100,random_state=42)
       model.fit(x,y)
       feature_importances=model.feature_importances_
       threshold=0.1
       selected_features = x.columns[feature_importances>threshold]
       print("Selected Features:")
       print(selected_features)

![388517612-afe8c286-877d-41f4-9392-3aa16803f600](https://github.com/user-attachments/assets/679167ab-784e-435a-b3fa-ac5e10221c88)

       from sklearn.ensemble import RandomForestClassifier
       model=RandomForestClassifier(n_estimators=100,random_state=42)
       model.fit(x,y)
       feature_importances=model.feature_importances_
       threshold=0.15
       selected_features = x.columns[feature_importances>threshold]
       print("Selected Features:")
       print(selected_features)

![388517967-dca07307-d7fc-4546-8614-fadbeb133ca0](https://github.com/user-attachments/assets/6e3e1e53-7211-4bc2-a5cc-320de4a870c3)

# RESULT:
Thus,Feature selection and Feature scaling has been used on the given dataset.
