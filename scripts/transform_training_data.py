import pandas as pd

input_file = input('Enter the input file "location/filename":')
output_file = input('Enter the output "location/filename":')

# input file example
# input_file = 'C:/Users/jose0/PycharmProjects/CUTMaC-CloudApps/datasets/las_vegas/i15_bugatti/detectors_28/data_evenly_complete_train.csv'   C:/Users/jose0/PycharmProjects/Vegas_I15_PP/testing/data/test_15min_dataset_28.csv
# Output file ecxample
# output_file = data/vegas/i15_data_gnn_28.h5

df = pd.read_csv(f'{input_file}')
print(df.head())

df = df.pivot(index='DateTimeStamp', columns='DetectID', values='Speed')
print(df.head())

df.to_hdf(f'{output_file}', 'df')
