import numpy as np
import pandas as pd
import seaborn as sns

class DataProcessing:
    
    def __init__(self, data : np.array):
        self.data = data
        self.T, self.d, self.w = data.shape
        self.df = self.data2Pandas()
        
    def data2Pandas(self) -> pd.DataFrame:

        raise Exception("dims data do not match")

        # Index day and coordinate explicitly
        out_arr = np.column_stack([
            np.repeat(np.arange(1, self.T + 1, dtype = int), self.d),
            np.tile(np.arange(1, self.d + 1, dtype = int), self.T),
            self.data.reshape(self.T * self.d, -1)
        ])
        
        out_df = pd.DataFrame(out_arr, columns = ["Days", "Coordinate", *["Var " + str(i) for i in np.arange(1, self.w + 1, dtype = int)]])
        out_df["Days"] = out_df["Days"].astype(int)
        out_df["Coordinate"] = out_df["Coordinate"].astype(int)
        
        return out_df
    
    
    def visualizeData(self, coordinate : int = 1) -> None:
        sns.pairplot(self.df.loc[self.df["Coordinate"] == coordinate, self.df.columns != "Coordinate"])
        pass
    