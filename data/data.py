import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

PATH = ""
X_train = pd.read_csv(PATH + "train.csv")
X_test = pd.read_csv(PATH + "test.csv")

X_train = X_train.drop(columns = ["OutcomeSubtype","AnimalID"])
X_test = X_test.drop(columns = ["ID"])

X_train, y_train = X_train.drop(columns = ["OutcomeType"]),X_train["OutcomeType"]
X_train = X_train.drop(columns = ["Color","Name","DateTime"])
X_test = X_test.drop(columns = ["Color","Name","DateTime"])
##Enlever les NAN
# print((X_train["AgeuponOutcome"].value_counts()/len(X_train))[:10])
Animal_imputer = SimpleImputer(strategy = 'constant', fill_value = 'Dog')
Sex_imputer = SimpleImputer(missing_values="Unknown",strategy = 'constant', fill_value = 'Neutered Male')
Sex_imputer1 = SimpleImputer(strategy = 'constant', fill_value = 'Neutered Male')
Age_imputer = SimpleImputer(strategy = 'constant', fill_value = '2 years')
Breed_imputer = SimpleImputer(strategy = 'constant', fill_value = 'Domestic Shorthair Mix')

#Train set
X_train["AnimalType"] = Animal_imputer.fit_transform(X_train["AnimalType"].values.reshape(-1,1))
X_train["SexuponOutcome"] = Sex_imputer.fit_transform(X_train["SexuponOutcome"].values.reshape(-1,1))
X_train["SexuponOutcome"] = Sex_imputer1.fit_transform(X_train["SexuponOutcome"].values.reshape(-1,1))
X_train["AgeuponOutcome"] = Age_imputer.fit_transform(X_train["AgeuponOutcome"].values.reshape(-1,1))
X_train["Breed"] = Breed_imputer.fit_transform(X_train["Breed"].values.reshape(-1,1))

#Test set
X_test["AnimalType"] = Animal_imputer.fit_transform(X_test["AnimalType"].values.reshape(-1,1))
X_test["SexuponOutcome"] = Sex_imputer.fit_transform(X_test["SexuponOutcome"].values.reshape(-1,1))
X_test["SexuponOutcome"] = Sex_imputer1.fit_transform(X_test["SexuponOutcome"].values.reshape(-1,1))
X_test["AgeuponOutcome"] = Age_imputer.fit_transform(X_test["AgeuponOutcome"].values.reshape(-1,1))
X_test["Breed"] = Breed_imputer.fit_transform(X_test["Breed"].values.reshape(-1,1))

#One Hot encoder for Dog Cat

animalType_encoder = OneHotEncoder(categories = 'auto', sparse = False)

#Train set
new_columns_train = animalType_encoder.fit_transform(X_train["AnimalType"].values.reshape(-1,1))
X_train = X_train.drop(columns=["AnimalType"])
X_train = pd.concat([X_train,pd.DataFrame(new_columns_train,columns=["cat","dog"])], axis = 1)


#Test set
new_columns_test = animalType_encoder.transform(X_test["AnimalType"].values.reshape(-1,1))

X_test = X_test.drop(columns=["AnimalType"])
X_test = pd.concat([X_test,pd.DataFrame(new_columns_test,columns=["cat","dog"])], axis = 1)

#AgeuponOutcome Split neutered and Sex
def parse_neutered(text):
    neutered, _ = text.split(" ")
    if (neutered == "Spayed"):
        neutered = "Neutered"
    return neutered

def parse_sex(text):
    _, sex = text.split(" ")
    return sex
# Train set
neutered_train = X_train.apply(lambda row: pd.Series(parse_neutered(row["SexuponOutcome"])), axis = 1  )
sex_train = X_train.apply(lambda row: pd.Series(parse_sex(row["SexuponOutcome"])), axis = 1  )

neutered_train.columns = ["neutered"]
sex_train.columns = ["sex"]

new_columns = pd.concat([neutered_train,sex_train], axis = 1)
X_train = X_train.drop(columns = ["SexuponOutcome"])
X_train = pd.concat([X_train,pd.DataFrame(new_columns)], axis = 1)

# Test set
neutered_test = X_test.apply(lambda row: pd.Series(  parse_neutered(row["SexuponOutcome"])  ), axis = 1  )
sex_test = X_test.apply(lambda row: pd.Series(  parse_sex(row["SexuponOutcome"])  ), axis = 1  )

neutered_test.columns = ["neutered"]
sex_test.columns = ["sex"]

new_columns = pd.concat([neutered_test,sex_test], axis = 1)
X_test = X_test.drop(columns = ["SexuponOutcome"])
X_test = pd.concat([X_test,pd.DataFrame(new_columns)], axis = 1)

#Label neutered and sex:
from sklearn.preprocessing import LabelEncoder

# encoder initialisation
neutered_label = LabelEncoder()
sex_label = LabelEncoder()


#Train set
X_train["neutered"] = neutered_label.fit_transform(X_train["neutered"].values)
X_train["sex"] = sex_label.fit_transform(X_train["sex"].values)

#Test set
X_test["neutered"] = neutered_label.transform(X_test["neutered"].values.reshape(-1,1))
X_test["sex"] = sex_label.transform(X_test["sex"].values.reshape(-1,1))

# print(X_train.head())

#Normaliser Age
def convert_months(text):
    age, label = text.split(" ")
    if (label == "year" or label == "years"):
        return int(age)*52.1429
    if (label == "month" or label == "months"):
        return int(age)*4.34524
    if (label == "week" or label == "weeks"):
        return int(age)

age_train = X_train.apply(lambda row: pd.Series(  convert_months(row["AgeuponOutcome"])  ), axis = 1  )
age_train.columns = ["Age"]

X_train = X_train.drop(columns = ["AgeuponOutcome"])
X_train = pd.concat([X_train,pd.DataFrame(age_train)], axis = 1)

# Test set
age_test = X_test.apply(lambda row: pd.Series(  convert_months(row["AgeuponOutcome"])  ), axis = 1  )
age_test.columns = ["Age"]

X_test = X_test.drop(columns = ["AgeuponOutcome"])
X_test = pd.concat([X_test,pd.DataFrame(age_test)], axis = 1)

print(X_train.head())

# # Scalers initialisation
age_scaler = StandardScaler()

#Train set
X_train["Age"] = age_scaler.fit_transform(X_train["Age"].values.reshape(-1,1))

#Test set
X_test["Age"] = age_scaler.transform(X_test["Age"].values.reshape(-1,1))

print(X_train.head())
