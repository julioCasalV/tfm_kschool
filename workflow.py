import pandas as pd
import numpy as np
import itertools
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

class Workflow():
    def __init__(self, n_neighbors = 15):
        self.model = None
        self.train = None
        self.n_neighbors = n_neighbors
    
    def etl(self, data: pd.DataFrame):
        dummy_columns = ['Bajo','Chalet','Piso']
        list_bajo = ["Entreplanta", "Uso comercial"]
        data.loc[data["Tipo"].isin(list_bajo), "Tipo"] = "Bajo"
        data.loc[~data["Tipo"].isin(dummy_columns), "Tipo"] = "Piso"
        dummy_type = pd.get_dummies(data['Tipo'])
        for column in dummy_columns:
            if column not in dummy_type.columns:
                dummy_type[column] = 0
        dummy_type = dummy_type[dummy_columns]
        data = pd.concat([data, dummy_type], axis=1)
        data = data.select_dtypes(exclude=object).drop('ID', axis = 1)
        return data
        
    def fit(self, file_data: str, conditions_sample = {'Latitud':0.1,'Longitud':0.1,'m2':0.25},
            file_factor_list: str = None):
        self.conditions_sample = conditions_sample
        data = pd.read_csv(file_data)
        self.data_orig = data
        data = self.etl(self.data_orig)
        if file_factor_list is None:
            self.factors_values = factors_values_initial(data)
        else:
            self.factors_values = pickle.load(open(file_factor_list, "rb"))
        self.predict_column = 'Precio_m2'
        self.fetures_exclude = ['Precio'] + [self.predict_column]
        self.features_column = [column for column in data if column not in self.fetures_exclude]
        self.data_nr = normalize(data, self.factors_values)

        if file_factor_list is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_nr[self.features_column], self.data_nr[self.predict_column], 
                                                                test_size = 0.3, random_state=42)
            factor_list = {}
            factor_list['m2'] = [0.5,1]
            factor_list['Tipo'] = [1]
            combinations = ranking_factors_combinations(self.X_train, self.X_test, factor_list, conditions_sample, self.y_train)
            self.factor_dict = combinations[0]
            self.factor_dict.pop("score")
        
            for column in self.factor_dict:
                self.factors_values.loc[column, "weight"] = self.factor_dict[column]
            self.data_nr = normalize(data, self.factors_values)
            pickle.dump(self.factors_values, open("factors_values", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_nr[self.features_column], self.data_nr[self.predict_column], 
                                                                test_size = 0.3, random_state=42)
        self.model=KNeighborsRegressor(n_neighbors = self.n_neighbors, weights = 'distance', metric = 'euclidean', 
                                       algorithm = 'brute')
        self.model.fit(self.X_train,self.y_train)
        
    def predict(self, X_test: pd.DataFrame):
        X_test = self.etl(X_test)
        X_test = X_test[self.features_column]
        X_test = normalize(X_test, self.factors_values)
        (neighbor_distance, neighbor_positions) = self.model.kneighbors(X_test)
        dict_result = {}
        for item, neighbor_list in enumerate(neighbor_positions):
            score_conditions = calculate_score_item(X_test.iloc[item], self.X_train.iloc[neighbor_list], self.conditions_sample)
            result = self.data_orig.loc[self.X_train.iloc[neighbor_list].index]
            for column in score_conditions:
                result[f'{column}_in'] = score_conditions[column].astype(int)
            dict_result[item] = {'item' : X_test.iloc[item], 'results' : result}
        # Vecinos calculados
        return dict_result

def factors_values_initial(data):
    factors_values = data.agg(['max', 'min']).transpose()
    factors_values['denom_factor'] = factors_values['max'] - factors_values['min']
    factors_values['weight'] = 1
    return factors_values

def normalize_column(df: pd.Series, factors_values):    
    return (df-factors_values['min'])/factors_values['denom_factor'] * factors_values['weight']

def normalize(df: pd.DataFrame, factors_values):
    df2 = df.copy()
    for column in df:
        df2[column] = normalize_column(df2[column], factors_values.loc[column])
    return df2

# X es el dataset que queremos comprobar, data es el dataset donde comprobamos (el de entrenamiento y el nuevo concatenado)
def parse_tipo(conditions: dict):
    list_tipo = ['Bajo','Chalet','Piso']
    if "Tipo" in conditions.keys():
        for item in list_tipo:
            conditions[item] = conditions["Tipo"]
        conditions.pop("Tipo")
    return conditions

def calculate_score_item(row: pd.Series, data: pd.DataFrame, conditions: dict):
    return (data[conditions.keys()] - row[conditions.keys()]).abs() < conditions.values()

def calculate_score(row: pd.Series, data: pd.DataFrame, conditions: dict):
    return ((data[conditions.keys()] - row[conditions.keys()]).abs() < conditions.values()).all(axis=1).mean()


def testigos_correctos(x:pd.DataFrame, data:pd.DataFrame, model, conditions = {'Latitud':0.1,'Longitud':0.1,'m2':0.25}): 
    
    # Añadimos o restamos a 1 el valor de porcentaje elegido para establecer testigos válidos en esos campos
    neighbors = model.kneighbors(x, return_distance=False)
    
    x["neighbors"] = list(neighbors)
    x["score"] = x.apply(lambda row: calculate_score(row.drop("neighbors"), data.iloc[row["neighbors"]], conditions), axis=1)
    
    return np.mean(list(x["score"]))

def total_factors_combinations(factor_list):
    list_tipo = ['Bajo','Chalet','Piso']
    total_factors_combinations = []
    factors_combinations = itertools.product(*factor_list.values())
    for factors_combination in factors_combinations:
        factors_item = {}
        for column, factor in zip(factor_list, factors_combination):
            if column == "Tipo":
                for column_tipo in list_tipo:
                    factors_item[column_tipo] = factor
            else:
                factors_item[column] = factor
        total_factors_combinations.append(factors_item)
    return total_factors_combinations

def ranking_factors_combinations(X_train, X_test, factor_list, conditions_sample, y_train):
    conditions_sample = parse_tipo(conditions_sample)
    factors_combinations = total_factors_combinations(factor_list)
    X_train_params = X_train.copy()
    X_test_params = X_test.copy()
    for factors_dict in factors_combinations:
        for column in factors_dict:
            X_train_params[column] = X_train[column]*factors_dict[column]
            X_test_params[column] = X_test[column]*factors_dict[column]
        knn = KNeighborsRegressor(n_neighbors = 15, weights = 'distance', metric = 'euclidean', algorithm = 'brute')
        knn.fit(X_train_params,y_train)
        score = testigos_correctos(X_test_params.copy(), X_train_params, knn, conditions_sample)
        factors_dict["score"] = score
    factors_combinations = sorted(factors_combinations, key=lambda item: item["score"], reverse=True)
    return factors_combinations