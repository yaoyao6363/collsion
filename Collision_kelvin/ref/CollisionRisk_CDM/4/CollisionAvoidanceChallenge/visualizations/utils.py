import enum
import os

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.manifold import TSNE

# Set the following property: plotly.io.orca.config.executable = '/path/to/orca'
ORCA_PATH = ''
plotly.io.orca.config.executable = ORCA_PATH
RISK = 'risk'
EVENT_ID = 'event_id'
C_OBJECT_TYPE = 'c_object_type'
MISSION_ID = 'mission_id'
THRESHOLD = -6


def tsne(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce dimensionality of dataframe, (not counting event_id, risk and c_object_type).
    :param df: Dataframe.
    :return: New reduced dataframe.
    """
    assert df.to_numpy().ndim == 2
    labels = pd.DataFrame({EVENT_ID: df[EVENT_ID], RISK: df[RISK]})
    df.drop(columns=[EVENT_ID, RISK], inplace=True)
    model = TSNE(n_components=2)
    data = model.fit_transform(df.to_numpy())
    new_df = pd.DataFrame(data=data, columns=['C1', 'C2'])
    new_df = pd.concat([labels, new_df], axis=1)
    return new_df


def standarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dataframe, (only event_id is not normalized).
    :param df: Dataframe.
    :return: New standardized dataframe.
    """
    assert df.to_numpy().ndim == 2
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(df.to_numpy()[:, 1:])
    new_df = pd.DataFrame(
        data=np.hstack((np.expand_dims(df.to_numpy()[:, 0], -1), data)),
        columns=df.columns)
    return new_df


def min_max(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dataframe, (only event_id and risk is not normalized).
    :param df: Dataframe.
    :return: New normalized dataframe.
    """
    assert df.to_numpy().ndim == 2
    labels = pd.DataFrame({EVENT_ID: df[EVENT_ID], RISK: df[RISK]})
    df.drop(columns=[EVENT_ID, RISK], inplace=True)
    scaler = preprocessing.MinMaxScaler()
    data = scaler.fit_transform(df.to_numpy())
    new_df = pd.DataFrame(data=data, columns=df.columns)
    new_df = pd.concat([labels, new_df], axis=1)
    return new_df


class ObjectTypes(enum.Enum):
    DEBRIS = 0
    PAYLOAD = 1
    ROCKET_BODY = 2
    TBA = 3
    UNKNOWN = 4


def create_df(path: str):
    """
    Create new dataframe with missing values changed to mean of the column.
    :param path: Path to read the dataframe.
    :return: None, saving new dataframe to the directory containing previous dataframe.
    """
    out_dir = os.path.dirname(path)
    df = pd.read_csv(path)
    df.drop(columns=[MISSION_ID], inplace=True)
    df = pd.get_dummies(df, prefix=[C_OBJECT_TYPE])
    df = df.fillna(df.mean())
    df.to_csv(os.path.join(out_dir, 'train_data_corrected.csv'), index=False)


def average_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each event average all CDM's.
    :param df: Dataframe containing events.
    :return: Averaged dataframe.
    """
    assert not df.isnull().values.any(), 'Data contain missing values.'
    data = np.zeros((np.unique(df[EVENT_ID].to_numpy()).size, df.shape[-1]))
    for i in np.unique(df[EVENT_ID].to_numpy()):
        event = df.loc[df[EVENT_ID] == i].to_numpy()
        data[i] = np.mean(event, axis=0)
    return pd.DataFrame(data=data, columns=df.columns)


def threshold_df(df: pd.DataFrame) -> pd.DataFrame:
    df[RISK][df[RISK] >= THRESHOLD] = 1
    df[RISK][df[RISK] < THRESHOLD] = 0
    return df


def show_parcoords(df: pd.DataFrame, path: str, max_samples: int = 100,
                   format_: str = 'png'):
    df = threshold_df(df)
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=df[RISK][:max_samples],
                  colorscale=[[0, 'red'], [1, 'blue']]),
        dimensions=[dict(label=str(key),
                         values=df[key][:max_samples]) for key in
                    df.drop(columns=[RISK, EVENT_ID]).keys()]))
    fig.update_layout(width=1500, height=1500)
    fig.show(renderer=format_)
    fig.write_image(
        os.path.join(path, 'parallel_coordinates.{}'.format(format_)))


def show_scatter(df: pd.DataFrame, path: str, format_='png'):
    df = threshold_df(df)
    fig = go.Figure(data=go.Scatter(x=df['C1'], y=df['C2'], mode='markers',
                                    marker_color=df[RISK],
                                    marker=dict(
                                        colorscale=[[0, 'red'], [1, 'blue']],
                                        line_width=1, size=15)))
    fig.update_layout(width=1500, height=1500,
                      xaxis=go.layout.XAxis(
                          title=go.layout.xaxis.Title(
                              text='Component 1',
                              font=dict(family="Courier New", size=20,
                                        color='#000000'))),
                      yaxis=go.layout.YAxis(
                          title=go.layout.yaxis.Title(
                              text='Component 2',
                              font=dict(family='Courier New', size=20,
                                        color='#000000'))))
    fig.show(renderer=format_)
    fig.write_image(os.path.join(path, 'scatter.{}'.format(format_)))


def show_hist(df: pd.DataFrame, path: str, format_: str = '.png'):
    fig = px.histogram(df, x=RISK)
    fig.write_image(os.path.join(path, 'histogram.{}'.format(format_)))
