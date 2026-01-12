from config import config
from mylogging import logging

import numpy as np
import pandas as pd
import json

num = int(config.parser.get('Core', 'num'))
value = config.parser.get('Core', 'value')
assist = config.parser.get('Core', 'assist')

def logic_fun(rabbitmq_consumer, rabbitmq_producer, sender_client, receiver_client,
              window_width, window_step, frequency):
    # get data by rabbitmq consumer
    if not rabbitmq_producer.connected:
        raise RuntimeError('RabbitMQ producer connecting to RabbitMQ broker')
    if not rabbitmq_consumer.connected:
        raise RuntimeError('RabbitMQ consumer connecting to RabbitMQ broker')
    logging.debug('Getting buffer from RabbitMQ broker...')
    
    buffer_full, buffer = rabbitmq_consumer.get_buffer()
    if not buffer_full and num == 0:
        # logging.warning('RabbitMQ consumer buffer not full!')
        raise RuntimeError('RabbitMQ consumer buffer not full, waiting...')
    time_stamp, feature = preprocess(buffer, frequency)
    record = {'measurement':'features', 'fields':{'feature':feature}, 'time':time_stamp}
    # send data by rabbitmq producer
    rabbitmq_producer.publish(json.dumps(record, default=str))
    logging.info(f'time {time_stamp} sent')






# ---------------------------------------------------------------- DETAILS --------------------------------
# filter out a window of data frame from the data frame streaming: discrete or continuous
def filter_df(df, df_assist, is_discrete, frequency):
    if not is_discrete:
        return filter_df_continuous(df, df_assist, frequency)
    else:
        return filter_df_discrete(df, df_assist, frequency)
# continuous
def filter_df_continuous(df, df_assist, frequency):
    start = df.iloc[0]['_start']
    end = df.iloc[0]['_stop']
    freq = frequency
    return df, start, end, freq
# discrete: find out the df of the LATEST non-zero feature
def filter_df_discrete(df, df_assist, frequency):
    start = None
    end = None
    freq = None

    try:
        # value df
        x1 = np.array(df['_time'])
        y1 = np.array(df['_value'])
        # assist df
        x2 = np.array(df_assist['_time'])
        y2 = np.array(df_assist['_value'])
        # find all possible start and end indices of non zero features
        split = []
        start_idxs = np.array([]).astype(int)
        start_idxs = np.concatenate((start_idxs, np.where((y2 == 0)[:-1] * (y2[1:] > 0))[0] + 1))
        end_idxs = np.array([]).astype(int)
        end_idxs = np.concatenate((end_idxs, np.where((y2 > 0)[:-1] * (y2[1:] == 0))[0]))
        # find the start and end index of the LASTEST non-zero feature
        if len(end_idxs) > 0 and len(start_idxs) == len(end_idxs):
            start_idx_ = start_idxs[-1]
            end_idx_ = end_idxs[-1]
            if start_idx_ < end_idx_:
                split = np.array([start_idx_, end_idx_])
                
                start_idx = split[0]
                end_idx = split[1]
                df = df.iloc[start_idx:end_idx]

                start = x1[start_idx] # new_feature_start = x1_orig[start]
                end = x1[end_idx] # new_feature_end = x1_orig[end]
                freq = (end - start) / num
        # scale feature
        feature_max = df['_value'].max()
        feature_min = df['_value'].min()
        df.loc[:, '_value'] = (df['_value'] - feature_min) / (feature_max - feature_min)
    except:
        pass

    return df, start, end, freq

def preprocess(_input: pd.DataFrame, frequency):
    # main df
    df = _input[_input['_tag']==value]
    df = df.reset_index(drop=True)
    # assist df
    df_assist = _input[_input['_tag']==assist]
    df_assist = df_assist.reset_index(drop=True)

    # find the start, end and freq
    is_discrete = True if num > 0 else False
    df, start, end, freq = filter_df(df, df_assist, is_discrete, frequency)

    if start is None or end is None or freq is None:
        raise ValueError(f"One of start ({start}), end ({end}) or freq ({freq}) is None!")
    
    # interpolation of feature with evenly spaced intervals
    df.set_index('_time', inplace=True)
    new_time_index = pd.date_range(
        start=start, end=end, freq=freq, inclusive='right')
    df = df.groupby(new_time_index[new_time_index.searchsorted(
                df.index, side='left')]).mean(numeric_only=True)
    df = df.reindex(index=new_time_index)
    df = df.interpolate(method='linear', limit_direction='both')

    # scale feature between 0 and 1
    feature = df['_value']
    feature = feature.to_csv(columns=['_value'], header=False, index=False)

    # find timestamp
    time_stamp = df.index[0] + (df.index[-1] - df.index[0]) / 2

    return time_stamp, feature