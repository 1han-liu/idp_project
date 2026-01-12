# CUMSTOMIZATIO: DATA DESERIALIZATION
# Please see the comments above the function deserialize_body

import pika
import time
from datetime import datetime
from mylogging import logging
import pandas as pd
from line_protocol_parser import parse_line

class RabbitMQClient():
    def __init__(self, broker_ip, broker_port, backup_size, key='value'):
        self.broker_ip = broker_ip
        self.broker_port = broker_port
        self.key = key
        logging.getLogger("pika").propagate = False
        self.body = None
        self.backup = None
        self.backup_size = backup_size
        self.connected = False

    def connect(self, name, is_consumer=False):
        self.is_consumer = is_consumer
        params = pika.ConnectionParameters(self.broker_ip, self.broker_port, client_properties={
        'connection_name': name}, heartbeat=0)
        self.connected = False
        while not self.connected:
            try:
                self.connection = pika.BlockingConnection(params) # Connect to broker
                logging.info("RabbitMQ %s connected to RabbitMQ broker!" % ('consumer' if self.is_consumer else 'producer'))
                self.connected = True
                logging.getLogger("pika").setLevel(logging.WARNING)
            except: 
                time.sleep(0.1)

    def setup(self, exchange):
        self.exchange = exchange
        self.channel = self.connection.channel() # start a channel
        self.channel.exchange_declare(exchange=self.exchange, exchange_type='fanout')
        logging.info("RabbitMQ %s channel set up and exchange %s declared!" % ('consumer' if self.is_consumer else 'producer', self.exchange))
        if self.is_consumer:
            self.result = self.channel.queue_declare(queue='', exclusive=True)
            self.queue_name = self.result.method.queue
            self.channel.queue_bind(exchange=self.exchange, queue=self.queue_name)
            logging.info("RabbitMQ %s queue binded to exchange %s!" % ('consumer' if self.is_consumer else 'producer', self.exchange))

    def publish(self, body):
        self.channel.basic_publish(
            exchange=self.exchange, 
            routing_key='',
            body=body)
        logging.debug("RabbitMQ %s published message to exchange %s!" % ('consumer' if self.is_consumer else 'producer', self.exchange))

    def subscribe(self, window_width=None, frequency=None, data_format=None, topic=None):
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.__callback,
            auto_ack=True)
        logging.info("RabbitMQ %s subscribed to exchange %s!" % ('consumer' if self.is_consumer else 'producer', self.exchange))
        self.window_width = window_width
        self.frequency = frequency
        self.data_format = data_format
        self.topic=topic
    
    def consume(self):
        self.channel.start_consuming()

    def disconnect(self):
        self.connection.close()
        logging.info("RabbitMQ %s disconnected!" % ('consumer' if self.is_consumer else 'producer'))

    def concat_backup(self, body):
        self.backup = pd.concat([self.backup, body], ignore_index=True)
        self.backup = self.backup.sort_values(by='time')
        head = pd.Timestamp.utcnow() - self.backup_size * self.window_width
        self.backup = self.backup.loc[self.backup['time'] >= head]

################################## DATA DESERIALIZATION ##################################
    # Structure of deserialized body
    # {'value': float, 'time': utc: datetime.datetime}
    # Please convert the passed column name self.key in __init__ to 'value' if necessary
    def deserialize_body(self, body):
        if self.data_format == 'influx':
            return self.influx2df(body)
        # ADD NEW ELIF BRANCH FOR NEW DESERIALZATION METHOD BELOW
        return body

    # Add new deserialization method below
    def influx2df(self, body):
        body = [parse_line(b) for b in body.decode('utf-8').strip().split('\n')]
        if not self.topic is None:
            try:
                body = [b for b in body if b['tags']['topic'] == self.topic]
                key = self.key
            except:
                key = body[0]['tags']['id'].split(';')[1].split('=')[1]
        body = pd.DataFrame(body)
        body['time'] = pd.to_datetime(body['time'], utc=True)
        body['value'] = body['fields'].apply(lambda x: x[key])
        body['tag'] = pd.Series([key] * len(body))
        body = body[['value', 'time', 'tag']]
        return body
    # ADD NEW DESERIALIZATION METHOD HERE
################################## DATA DESERIALIZATION ##################################

    def __callback(self, ch, method, properties, body):
        if not body == '':
            self.body = self.deserialize_body(body)
            if not self.window_width is None or not self.frequency is None:
                if self.backup is None:
                    self.backup = self.body
                else:
                    self.concat_backup(self.body)
        
    def get_buffer(self):
        self.stop = pd.Timestamp.utcnow()
        self.start = self.stop - self.window_width
        buffer = self.backup.copy()
        buffer = buffer.loc[buffer['time'] >= self.start]
        if len(buffer) >= int(self.window_width / self.frequency):
            buffer_full = True
        else:
            buffer_full = False
        buffer['stop'] = self.stop
        buffer['start'] = self.start
        buffer.rename(
            columns={'time': '_time', self.key: '_' + self.key, 'tag': '_tag', 'start': '_start', 'stop': '_stop'}, 
            inplace=True)
        return buffer_full, buffer