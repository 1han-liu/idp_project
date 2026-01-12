# test processor logic
# 1. check rabittmq_consumer: telegraf producer (rabbitmq client) --> TEMPLATE_PROCESSOR_consumer
# 2. check rabittmq_producer: TEMPLATE_PROCESSOR_consumer --> TEMPLATE_PROCESSOR_producer
# 3. check influxdb sender: TEMPLATE_PROCESSOR_consumer --> influxdb_sender (sink)
# 4. check influxdb receicer: influxdb_sender --> influxdb_receiver
from config import config
from mylogging import logging

import numpy as np
import pandas as pd
import json

import onnx
import onnxruntime
from pathlib import Path


def logic_fun(rabbitmq_consumer, rabbitmq_producer, sender_client, receiver_client,
              window_width, window_step, frequency):
    if not rabbitmq_consumer.connected:
        raise RuntimeError('RabbitMQ consumer connecting to RabbitMQ broker')
    logging.debug('Getting buffer from RabbitMQ broker...')
    time_stamp, feature = parse(rabbitmq_consumer.body)
    loss = evaluate(feature)
    logging.info(f'time {time_stamp} loss {loss} calculated!')
    # persistent storage
    fields = {'loss':loss.item()}
    sender_client.write(fields, time_stamp)


# ---------------------------------------------------------------- DETAILS ----------------------------------------------------------------
# deserialization
def parse(input_data):
    dict_data = json.loads(input_data)
    time_stamp = dict_data['time']
    feature = np.fromstring(dict_data['fields']['feature'], sep='\n', dtype=np.float32)
    feature = np.reshape(feature, (1, -1))
    return time_stamp, feature


# model evaluation
onnx_model = onnx.load('autoencoder.onnx')
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession('autoencoder.onnx')

# two input names exists due to the default input name of tensorflow is "input_1" but we usually assign "input" as input name in "pytorch"
input_name = 'input'
other_input_name = 'input_1'

def evaluate(inputData):
    try:
        ort_input = {input_name: inputData}
        recon = ort_session.run(None, ort_input)
    except:
        ort_input = {other_input_name: inputData}
        recon = ort_session.run(None, ort_input)
    loss = np.mean((recon-inputData)**2, axis=2)
    return loss