import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import statistics
import random
import matplotlib.pyplot as plt

def keep_column(*,dataframe:pd.DataFrame,col:str):
    new_df = pd.DataFrame()
    new_df["Date"] = dataframe["Date"]
    new_df[col] = dataframe[col]
    return new_df

class TimeSeries:
    """Interface class to load all csv file from folder passed in argument"""
    def __init__(self,path):
        self.path = path
    def keys(self):
        """This method builds the dictionnary keys using a list

        Returns:
            list: list of the differents keys of the dictionnary
        """
        return [file for file in os.listdir(self.path)]

    def __getitem__(self, file):
        """This method is called in the initialization of the dictionnary
        It assign a value to all the different element of the list based on the element

        Args:
            file (string): this argument refers
                to the value in the list returned by the keys

        Returns:
            dataframe: dataframe loaded with the path provided in
                the class initialization combined with the csv files read in the directory
        """
        df = pd.read_csv(self.path+file)
        return keep_column(dataframe=df,col="Adj Close")



def yield_calculation(*,dataframe:pd.DataFrame):
    new_df = pd.DataFrame()
    for key,value in dataframe.items():
        new_df[key]= pd.DataFrame([
            (dataframe[key][row_number] / dataframe[key][row_number-1])
                    / dataframe[key][row_number-1]
            for row_number in range(len(dataframe[key]))
            if row_number != 0 and row_number !=len(dataframe[key])
        ],columns=["Yields"])["Yields"]
        
        average_return = new_df.mean()[0]
        standard_deviation = new_df.std()[0]
    return  (average_return,standard_deviation)




def merge_database_to_dataframe(*,database:dict):
    """This method merges all the dataframes of a database into one single dataframe
    based on the column "Date"
    """
    dataframe = pd.DataFrame()
    test = 0
    for key,value in database.items():
        mapper = {
            "Date":"Date"
        }
        for index,column_name in enumerate(value.columns.array):
            if index != 0:
                mapper[column_name]=column_name+" "+key

        value = value.rename(columns=mapper)
        if test == 0:
            dataframe = value
            test+=1
        else:
            dataframe = pd.merge(dataframe, value,on="Date")

    return dataframe

def rand_weights(number_of_assets):
    """Produces n random weights that sum to 1 """
    weights = np.random.rand(number_of_assets)
    return weights / sum(weights)


def random_portfolio(dataframe):
    """
    Returns a randomly generated portfolio from dataframes passed in argument
    """
    weights = rand_weights(dataframe.shape[1])
    portfolio = pd.DataFrame()
    for index,weight in enumerate(weights):
        if index==0:
            portfolio["portfolio"]=dataframe.iloc[:, index]*weight
        else:
            portfolio["portfolio"]=portfolio["portfolio"] + dataframe.iloc[:, index]*weight

    return portfolio








# Load all the datasets from the folder Database/ as pandas DataFrame into a dictionnary
database = dict(TimeSeries("Database/"))
#Merge all the DataFrames from the database into a single DataFrame
dataframe = merge_database_to_dataframe(database=database)
# Remove date from dataframe in order to create the dataframe with yields instead of values
dataframe = dataframe.drop(columns=["Date"])

number_of_portfolios = 5000
means, stds = np.column_stack([
    yield_calculation(dataframe = random_portfolio(dataframe))
    for _ in range(number_of_portfolios)
])

fig = plt.figure()
plt.plot(stds, means, 'o', markersize=1)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')
fig.savefig("Figure.png")
