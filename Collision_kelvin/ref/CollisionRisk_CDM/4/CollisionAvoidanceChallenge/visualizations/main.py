import pandas as pd

from visualizations import utils

if __name__ == '__main__':
    PATH = ''
    DATA_PATH = ''
    utils.create_df(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df = utils.min_max(df)
    df = utils.average_events(df)
    # utils.show_parcoords(pd.concat([df.iloc[:, :3], df.iloc[:, 30:40]], axis=1), PATH)
    utils.show_scatter(utils.tsne(df.iloc[:10000]), PATH)
    # df = utils.threshold_df(df)
    # utils.show_hist(df, PATH)
