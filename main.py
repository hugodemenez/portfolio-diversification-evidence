"""Insppired by https://towardsdatascience.com/python-markowitz-optimization-b5e1623060f5
"""

# Standard imports
import os

# Other imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        return [file for file in os.listdir(f"./input/{self.path}")]

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
        dataframe = pd.read_csv(f"./input/{self.path}{file}")
        return keep_column(dataframe=dataframe,col="Adj Close")



def yield_calculation(*,dataframe:pd.DataFrame):
    new_df = pd.DataFrame()
    for key,_ in dataframe.items():
        new_df[key]= pd.DataFrame([
            (dataframe[key][row_number] / dataframe[key][row_number-1])
                    / dataframe[key][row_number-1]
            for row_number in range(len(dataframe[key]))
            if row_number != 0 and row_number !=len(dataframe[key])
        ],columns=["Yields"])["Yields"]

        average_return = new_df.mean()[0]
        standard_deviation = dataframe.std()[0]
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

    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    # We only keep data after 2013 and before 2018
    dataframe = dataframe[~(dataframe['Date'] < '2015-01-01')]
    dataframe = dataframe[~(dataframe['Date'] > '2020-01-01')]

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


def draw_correlation_matrix(*, dataframe: pd.DataFrame, directory:str):
    """Draws the correlation matrix among the provided dataframe

    Args:
        dataframe (pd.DataFrame): DataFrame containing the analyzed datasets
        directory (str): _description_
    """
    # Show correlation matrix
    figure, canvas = plt.subplots()
    sns.heatmap(dataframe.corr(method='pearson'), annot=True, fmt='.4f',
            cmap=plt.get_cmap('coolwarm'), cbar=False, ax=canvas)
    canvas.set_yticklabels(canvas.get_yticklabels(), rotation="horizontal")
    figure.savefig(
        f"./output/{directory}_correlation_matrix.png",
        bbox_inches='tight',
        pad_inches=0.0,
    )



def draw_markowitz_border(*,directory:str,batch_size:int = 1000):
    """Draws the Markowitz border through a simulation of porfolio built 
    based on data hosted in the directory passed in argument

    The Markowitz drawing is saved as the name of  {directory_name}_markowitz.png

    Args:
        directory (str): Directory hosting datasets (obtained from Yahoo Finance)
        batch_size (int, optional): Number of portfolio simulation. Defaults to 50000.
    """
    # Load all the datasets from the folder Database/ as pandas DataFrame into a dictionnary
    database = dict(TimeSeries(directory+"/"))
    #Merge all the DataFrames from the database into a single DataFrame
    dataframe = merge_database_to_dataframe(database=database)
    # Remove date from dataframe in order to create the dataframe with yields instead of values
    dataframe = dataframe.drop(columns=["Date"])
    # Compute the logaritmic return for each dataset
    returns = (dataframe/dataframe.shift(1)).sub(1)
    # Draws the correlation matrix for returns in DataFrame
    draw_correlation_matrix(dataframe=returns,directory=directory)

    # Defining arrays
    all_weights = np.zeros((batch_size, len(dataframe.columns)))
    ret_arr = np.zeros(batch_size)
    vol_arr = np.zeros(batch_size)
    sharpe_arr = np.zeros(batch_size)

    # Generating portfolios
    np.random.seed(22)
    for x in range(batch_size):
        # Weights
        weights = np.array(np.random.random(len(dataframe.columns)))
        weights = weights/np.sum(weights)

        # Save weights
        all_weights[x,:] = weights

        # Expected return
        ret_arr[x] = np.sum( (returns.mean() * weights)*252)

        # Expected volatility
        vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))

        # Sharpe Ratio
        sharpe_arr[x] = ret_arr[x]/vol_arr[x]


    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]


    fig = plt.figure(figsize=(12,8))
    plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title(f"Markowitz border for {directory} indexes")
    plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50,marker="x") # red dot

    #Creates the output folder if it doesn't exist
    if not os.path.exists("./output/"):
        os.makedirs("output")

    fig.savefig(f"./output/{directory}_markowitz.png")


draw_markowitz_border(directory="Combined")
draw_markowitz_border(directory="Developed")
draw_markowitz_border(directory="Emerging")
