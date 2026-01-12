import os
from typing import Dict, Optional

from influxdb import InfluxDBClient


class InfluxWriter:
    """
    Small helper around the InfluxDB 1.x client so the UI can persist captured
    coordinates (or any future measurement) without duplicating boilerplate.
    """

    def __init__(self) -> None:
        self.host = os.getenv("INFLUX_HOST", "influxdb")
        self.port = int(os.getenv("INFLUX_PORT", "8086"))
        self.username = os.getenv("INFLUX_USERNAME", "mldeploy")
        self.password = os.getenv("INFLUX_PASSWORD", "mldeploy")
        self.database = os.getenv("INFLUX_DATABASE", "input")
        self.enabled = os.getenv("INFLUX_DISABLE_WRITES", "0") != "1"
        self.client: Optional[InfluxDBClient] = None

    def connect(self) -> None:
        if self.client is None:
            self.client = InfluxDBClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                database=self.database,
            )

    def write_point(
        self,
        measurement: str,
        fields: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        if not self.enabled:
            return False
        self.connect()
        assert self.client is not None

        body = {
            "measurement": measurement,
            "fields": fields,
        }
        if tags:
            body["tags"] = tags
        if timestamp:
            body["time"] = timestamp
        return self.client.write_points([body])

    def delete_points(
        self,
        measurement: str,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Delete points by tags and optional exact timestamp.
        For reliability, prefer deleting by line_id/endpoint tags rather than time alone.
        """
        if not self.enabled:
            return False
        self.connect()
        assert self.client is not None
        conditions = []
        if tags:
            conditions.extend([f"{k} = '{v}'" for k, v in tags.items()])
        if timestamp:
            conditions.append(f"time = '{timestamp}'")
        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"DELETE FROM {measurement}{where_clause}"
        result = self.client.query(query)
        return bool(result)
