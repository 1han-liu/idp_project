import time

import numpy as np
import threading
import signal

import grpc
import proto.ProcessLogDataProvider_pb2_grpc as pdp
import proto.ProcessLogConfiguration_pb2 as pc
import proto.ProtoProcessLogEntryContainer_pb2 as pe

import asyncio
import socket

from config import config
from mylogging import logging

## CONFIGURATION
sensor_name = 1
sample_interval = float(config.get('SENSOR_' + str(sensor_name), 'sample_interval'))
send_interval = float(config.get('SENSOR_' + str(sensor_name), 'send_interval'))
pos = int(config.get('SENSOR_' + str(sensor_name), 'pos'))
buffer_size = int(config.get('SENSOR_' + str(sensor_name), 'buffer_size'))
tab = eval('pe.ProtoProcessLogEntryDataWert.' + config.get('SENSOR_' + str(sensor_name), 'tab_name'))
resolution_value = int(config.get('SENSOR_' + str(sensor_name), 'resolution'))
pos_list = [pos]
buffer = []

# stop event
stop_event = threading.Event()

# ---------------------------------------------------------------- START REQUEST: start the client ----------------------------------------------------------------
# set configuration for protobuf
def make_process_log_configuration():
    parameters = [
        pc.Parameter(
                tab=tab,
                pos=pos,
                resolutions=[
                    pc.Resolution(value=resolution_value)
                ]
            ) for pos in pos_list]
    process_log_configuration = pc.ProcessLogConfiguration(  
        parameters=parameters
    )
    return process_log_configuration
# make request
def request_process_log_data_stream(stub):
    containers = stub.RequestProcessLogDataStream(
        make_process_log_configuration())
    return containers
# start request: blocking
def start_request():
    ip_ad = 'host.docker.internal'
    port = 59001
    logging.info('gRPC request started...')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((ip_ad, port))
    while not result == 0:
        logging.warning('Port to Arburg-gRPC is not opened! Please open the Arburg-gRPC server!')
        time.sleep(0.2)
        result = sock.connect_ex((ip_ad, port))
    channel = grpc.insecure_channel(f'{ip_ad}:{port}') # Valid for windows and mac. For linux: Starting from version 20.10 , the Docker Engine now also supports communicating with the Docker host via host.docker.internal on Linux. Unfortunately, this won't work out of the box on Linux because you need to add the extra — add-hostrun flag: --add-host=host.docker.internal:host-gateway    
    stub = pdp.ProcessLogDataProviderStub(channel)
    containers = request_process_log_data_stream(stub)
    return containers
# ---------------------------------------------------------------- RUN GRPC CLIENT: retrieve the data from the container --------------------------------
# post porcess to list
def postprocess(container):
    global buffer
    arr = []
    for entry in container.entries:
        arr.append([
        *[data.protoWert.mFloat for data in entry.protoProcessLogData]])
    arr = np.array(arr)
    if len(buffer) == 0:
        buffer = arr
    else:
        buffer = np.vstack((np.array(buffer), arr))
    if len(buffer) > 0:
        buffer = buffer[-buffer_size:, :]
    buffer = buffer.tolist()
# request function with post processing: blocking
def do_request(containers):
    try:
        while not stop_event.is_set():
            postprocess(next(containers))
    except Exception as e:
        logging.error(e)
def run_grpc_client(containers):
    do_request(containers)
# ---------------------------------------------------------------- STOP REQUEST: stop the client ----------------------------------------------------------------
# stop request
def stop_request(containers):
    containers.cancel()
    logging.info('gRPC request cancelled!')
# ---------------------------------------------------------------- SAMPLING ----------------------------------------------------------------
def sample_func(time_stamp):
    global buffer
    if len(buffer) == 0:
        return
    sample_value = buffer.pop(0)[0]
    return time_stamp, sample_value

# set the stop event
def signal_handler(containers, grpc_client_thread):
    stop_event.set()


if __name__ == '__main__':
    stop_event.clear()
    # start the client to receive the stream and get the data from the streaming
    containers = start_request()
    grpc_client_thread = threading.Thread(target=lambda: run_grpc_client(containers))
    signal.signal(signal.SIGINT, lambda signal, frame: signal_handler(containers, grpc_client_thread))
    # run the client in background
    grpc_client_thread.setDaemon(True)
    grpc_client_thread.start()

    # sampling the data and send the data regularly
    output = []
    # sample in a loop
    async def sample_loop():
        async def sample_coro():
            time_stamp = time.time()
            try:
                time_stamp, sample_value = sample_func(time_stamp)
            except:
                return
            output.append({'time': time_stamp, 'value': sample_value})
            return
        while not stop_event.is_set():
            await asyncio.gather(asyncio.sleep(sample_interval), sample_coro())
    # send in a loop with larger interval
    async def send_loop():
        async def send_coro():
            # print all data using influxdb form
            if not len(output):
                return
            info = "\n ".join([f"grpc_server,topic=input value={item['value']} {int(item['time'] * 1e9)}" for item in output])
            output.clear()
            print(info, flush=True)
            return
        while not stop_event.is_set():
            await asyncio.gather(asyncio.sleep(send_interval), send_coro())
    # start the loop until stop event is set
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(asyncio.gather(sample_loop(), send_loop()))
    stop_request(containers)
    grpc_client_thread.join()

    