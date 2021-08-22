import argparse
import numpy as np
import os
import pandas as pd
import geopy.distance as geo

def generate_distance_file(args):
    df = pd.read_csv(args.location_df_filename)

    distances = pd.DataFrame(columns=['from','to','cost'])
    for index0, row0 in df.iterrows():
        cord0 = (df['latitude'][index0], df['longitude'][index0])
        for index1, row1 in df.iterrows():
            cord1 = (df['latitude'][index1], df['longitude'][index1])
            distances = distances.append(pd.DataFrame([[row0['sensor_id'], row1['sensor_id'], geo.distance(cord0, cord1).m]],
                                                      columns=['from', 'to', 'cost']))

    print(distances.head(30))

    distances.to_csv(args.output_dir + '/' + args.output_filename, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/sensor_graph", help="Output directory.")
    parser.add_argument("--output_filename", type=str, default="distance_vegas_28.csv", help="Output directory.")
    parser.add_argument("--location_df_filename", type=str, default="data/sensor_graph/graph_sensor_locations.csv", help="Raw traffic readings.",)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_distance_file(args)