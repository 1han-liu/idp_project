import sys
from config import config
from rabbitmq_client import RabbitMQClient
from mylogging import logging
from datetime import datetime
## CONFIGURATION for RabbitMQ
rabbitmq_broker_ip = config.parser.get('RabbitMQ', 'broker_ip')
rabbitmq_broker_port = config.parser.get('RabbitMQ', 'broker_port')
rabbitmq_backup_size = config.parser.get('RabbitMQ', 'backup_size')
rabbitmq_backup_size = float(rabbitmq_backup_size)
rabbitmq_producer_exchange = config.parser.get('RabbitMQ', 'producer_exchange')
# create a new RabbitMQ client
rabbitmq_producer = RabbitMQClient(rabbitmq_broker_ip, rabbitmq_broker_port, rabbitmq_backup_size)

# function for RabbitMQ producer
def rabbitmq_producer_run():
    # produce
    logging.info('RabbitMQ producer connection starts...')
    start = datetime.now()
    rabbitmq_producer.connect('telegraf_producer', is_consumer=False)
    rabbitmq_producer.setup(rabbitmq_producer_exchange)
    end = datetime.now()
    logging.info('RabbitMQ producer connection takes %ss' % (end - start).total_seconds())
# start the producer
rabbitmq_producer_run()
# publish by RabbitMQ producer frm stadard input in a loop
while True:
    try:
        line = sys.stdin.readline()
        rabbitmq_producer.publish(line)
    except: pass
