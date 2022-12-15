import pandas as pd

class DataClean:
    def __init__(self,df):
        self.df = df
    def Cleaner(self):
        self.df = pd.read_csv('cars_all_v3.csv')
        self.df = self.df.drop_duplicates(subset='link')
        self.df[['cc','hp']]= self.df['engine'].str.split('/',expand=True)
        self.df = self.df.drop('engine',axis=1)
        
        self.df['price'] = self.df.price.str.replace('.','')
        self.df['price'] = self.df.price.str.replace('\n','')
        self.df['mileage'] = self.df.mileage.str.replace('.','')
        self.df['mileage'] = self.df.mileage.str.replace('\n','')
        
        self.df = self.df[~(self.df.car_fuel=="Αλλο")]
        self.df['car_fuel'] = self.df['car_fuel'].replace("Αέριο(lpg)-Βενζίνη",'gas')
        self.df['car_fuel'] = self.df['car_fuel'].replace("Φυσικόαέριο(cng)",'gas')
        
        self.df['car_fuel'] = self.df.car_fuel.str.replace('ΥβριδικόΠετρέλαιο','hybrid')
        self.df['car_fuel'] = self.df.car_fuel.str.replace('ΥβριδικόΒενζίνη','hybrid')
        self.df['car_fuel'] = self.df.car_fuel.str.replace('Βενζίνη','gasoline')
        self.df['car_fuel'] = self.df.car_fuel.str.replace('Πετρέλαιο','diesel')
        self.df['car_fuel'] = self.df.car_fuel.str.replace('Ηλεκτρικό','electric')
        
        self.df = self.df.dropna()
        
        self.df['mileage'] = self.df['mileage'].astype(int)
        self.df['car_type'] = self.df['car_type'].astype("category")
        self.df['car_fuel'] = self.df['car_fuel'].astype("category")
        self.df['maker'] = self.df['maker'].astype("category")
        self.df['model'] = self.df['model'].astype("category")
        self.df['transmission'] = self.df['transmission'].astype("category")
        self.df['year'] = self.df['year'].astype(int)
        self.df['cc'] = self.df['cc'].astype(int)
        self.df['hp'] = self.df['hp'].astype(int)
        
        self.df.to_csv('test.csv')
        print(self.df)