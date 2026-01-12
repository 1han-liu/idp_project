from config import config
from mylogging import logging
from rabbitmq_client import RabbitMQClient
from my_influxdb_client import MyInfluxDBClient
from logic import logic_fun

import threading
import pandas as pd
from datetime import datetime
import time
import json
from func_timeout import func_set_timeout
import func_timeout


# general
name = config.parser.get('General', 'name')
window_width = config.parser.get('General', 'window_width')
window_step = config.parser.get('General', 'window_step')
frequency = config.parser.get('General', 'frequency')
window_width = pd.to_timedelta(window_width)
window_step = pd.to_timedelta(window_step)
frequency = pd.to_timedelta(frequency)

# RabbitMQ client - general
rabbitmq_broker_ip = config.parser.get('RabbitMQ', 'broker_ip')
rabbitmq_broker_port = config.parser.get('RabbitMQ', 'broker_port')
rabbitmq_backup_size = config.parser.get('RabbitMQ', 'backup_size')
rabbitmq_backup_size = float(rabbitmq_backup_size)
# RabbitMQ client - consumer
rabbitmq_consumer_exchange = config.parser.get('RabbitMQ', 'consumer_exchange')
rabbitmq_consumer_data_format = config.parser.get('RabbitMQ', 'consumer_data_format')
rabbitmq_consumer = RabbitMQClient(rabbitmq_broker_ip, rabbitmq_broker_port, rabbitmq_backup_size)
# RabbitMQ client - producer
rabbitmq_producer_exchange = config.parser.get('RabbitMQ', 'producer_exchange')
rabbitmq_producer = RabbitMQClient(rabbitmq_broker_ip, rabbitmq_broker_port, rabbitmq_backup_size)

# influxdb client - general
url = config.parser.get('InfluxDB', 'url')
username = config.parser.get('InfluxDB', 'username')
password = config.parser.get('InfluxDB', 'password')
token = f'{username}:{password}'
# influxdb client - receiver
receiver_database = config.parser.get('InfluxDB', 'receiver_database')
receiver_measurement = config.parser.get('InfluxDB', 'receiver_measurement')
receiver_client = MyInfluxDBClient(url=url, token=token, database=receiver_database, measurement=receiver_measurement)

# influxdb client - sender
sender_database = config.parser.get('InfluxDB', 'sender_database')
sender_measurement = config.parser.get('InfluxDB', 'sender_measurement')
sender_client = MyInfluxDBClient(url=url, token=token, database=sender_database, measurement=sender_measurement)

# ---------------------------------------------------------------- LOGIC ----------------------------------------------------------------
# test processor logic
def processor_logic():
    logic_fun(rabbitmq_consumer, rabbitmq_producer, sender_client, receiver_client,
              window_width, window_step, frequency)
# ---------------------------------------------------------------- LOGIC ----------------------------------------------------------------

def rabbitmq_consumer_run():
    # consume
    logging.info('RabbitMQ cosumer connection starts...')
    start = datetime.now()
    rabbitmq_consumer.connect(f'{name}_consumer', is_consumer=True)
    rabbitmq_consumer.setup(rabbitmq_consumer_exchange)
    rabbitmq_consumer.subscribe(window_width, frequency, rabbitmq_consumer_data_format, rabbitmq_consumer_exchange)
    end = datetime.now()
    logging.info('RabbitMQ consumer connection takes %ss' % (end - start).total_seconds())
    rabbitmq_consumer.consume()

def rabbitmq_producer_run():
    # produce
    logging.info('RabbitMQ producer connection starts...')
    start = datetime.now()
    rabbitmq_producer.connect(f'{name}_producer', is_consumer=False)
    rabbitmq_producer.setup(rabbitmq_producer_exchange)
    end = datetime.now()
    logging.info('RabbitMQ producer connection takes %ss' % (end - start).total_seconds())

def influxdb_receiver_run():
    receiver_client.create_query_api()

def influxdb_sender_run():
    sender_client.create_write_api()


@func_set_timeout(window_step.total_seconds())
def event():
    try:
        processor_logic()
    except RuntimeError as e:
        logging.error('Runtime error: %s' % e)
    except Exception as e:
        logging.error(e)

def core():
    while not stopEvent.is_set():
        try:
            event()
        except func_timeout.exceptions.FunctionTimedOut as e: 
            logging.error('Time out error: %s' % e)
        time.sleep(window_step.total_seconds())

# stop event
stopEvent = threading.Event()

# create threads
## rabibitmq consumer
rabbitmq_consumer_thread = threading.Thread(target=rabbitmq_consumer_run)
rabbitmq_producer_thread = threading.Thread(target=rabbitmq_producer_run)
## influxdb
influxdb_receiver_thread = threading.Thread(target=influxdb_receiver_run)
influxdb_sender_thread = threading.Thread(target=influxdb_sender_run)
# core part
core_thread = threading.Thread(target=core)

# start threads
def start():
    if eval(config.parser.get('RabbitMQ', 'consumer_activate')):
        rabbitmq_consumer_thread.start()
    if eval(config.parser.get('RabbitMQ', 'producer_activate')):
        rabbitmq_producer_thread.start()
    if eval(config.parser.get('InfluxDB', 'receiver_activate')):
        influxdb_receiver_thread.start()
    if eval(config.parser.get('InfluxDB', 'sender_activate')):
        influxdb_sender_thread.start()
    core_thread.start()
# stop threads
def stop():
    stopEvent.set()
    core_thread.join()
