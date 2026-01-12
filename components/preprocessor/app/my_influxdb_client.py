from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

class MyInfluxDBClient(InfluxDBClient):
    def __init__(self, url, token, database, measurement):
        super().__init__(url=url, token=token)
        self.database = database
        self.measurement = measurement

    def create_query_api(self):
        self.my_query_api =  self.query_api()
    
    def create_write_api(self):
        self.my_write_api = self.write_api(write_options=SYNCHRONOUS)
    
    def get_query(self, window_width, field):
        query = f'from(bucket: "{self.database}/autogen")\
            |> range(start: - {int(window_width.total_seconds() * 1e9)}ns)\
            |> filter(fn: (r) => r._measurement == "{self.measurement}")\
            |> filter(fn: (r) => r["_field"] == "{field}")\
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        res = self.my_query_api.query_data_frame(query)
        return res
    
    def write(self, fields, timestamp):
        record = {'measurement': self.measurement, 'fields': fields, 'time': timestamp}
        with self.my_write_api as _write_client:
            _write_client.write(f'{self.database}/autogen', 'wbk', record=record)
