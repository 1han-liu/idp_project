@echo off
@REM This script creates and sets up the Docker bind mount directories on the host machine as specified in the compose.yaml.
@REM The directories will be created in the folder docker-volumes\... insides this project folder and are ignored by source control (git).
@REM The folder "docker-volumes" and it's contents will be first deleted and then recreated, if it already exists. 

@REM copy files
if not exist ".\docker-volumes\grafana\grafana.db" xcopy ".\components\grafana\grafana.db" ".\docker-volumes\grafana\" 
if not exist ".\docker-volumes\influxdb\config\influxdb.conf" xcopy ".\components\influxdb\influxdb.conf" ".\docker-volumes\influxdb\config\"
if not exist ".\docker-volumes\telegraf\config\telegraf.conf" xcopy ".\components\telegraf\telegraf.conf" ".\docker-volumes\telegraf\config\" 
if not exist ".\docker-volumes\rabbitmq\config\rabbitmq.conf" xcopy ".\components\rabbitmq\rabbitmq.conf" ".\docker-volumes\rabbitmq\config\"

if not exist ".\docker-volumes\mosquitto\config\mosquitto.conf" xcopy ".\external\mosquitto\mosquitto.conf" ".\docker-volumes\mosquitto\config\"

echo "bind mount files set up"

@REM build and start containers
docker compose up -d --build