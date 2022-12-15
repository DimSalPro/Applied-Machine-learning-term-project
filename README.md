# Clustering  --> Dataset: https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990)

All the files along with the USCensus1990.data.txt must be on the same file/directory

modifier.py is a  function that lower the size of the data

Install all the necessary libraries ( pip install yellowbrick )

run Clustering_final_part1A.py

----------------------------------------------------------------
# Regression --> Dataset: Scrapped from car.gr

Contains 3 csv files which will automatically will be merged
Run the main.py file inside the regression folder

The scrapper will be out of order for now due to website's upgrade,
To run the scrapper got inside spider folder and on the command line type " scrapy crawl cars -o cars.csv"


----------------------------------------------------------------
# Classification --> Dataset: https://recsys2019data.trivago.com/

All the files along with the train.csv must be on the same file/directory


Run Classification_final_part3A.py 

This will produce a csv file of the final transformed dataset the the Classification_final_part3B.py will use


Before you run Classification_final_part3B.py, you will have to pick which method you want the algorithm to run

On the lines 282 - 285 are the 4 different functions that has been created

Put 1 to the one that you want to run and 0 for all the others (default is set to the model with the best results)

whole_dataset attribute runs for the whole dataset without SVM

sample_dataset runs on random sample of 1% incliding SVM

over_sampled_dataset runs on the whole dataset, which the minority class have been oversampled (without SVM)

under_sampled_dataset runs on 1% of the dataset ,which the majority class have been undersampled (with SVM)



