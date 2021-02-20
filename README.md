# Instructions
## Docker Image
1. Change to the docker directory
```console
$ cd Docker/pick_and_place
```
2. Build or update the docker image.
```console
$ docker build -t robottycoon.azurecr.io/unity-robotics-pick-and-place -f docker/Dockerfile .
```

3. Start the image.
```console
$ docker run -it --rm -p 10000:10000 -p 50051:50051 robottycoon.azurecr.io/unity-robotics-pick-and-place:alpha ./src/start.sh
```

Port 10000 is the port that ROS listens to for communication with the simulator. Port 50051 is the port that the GRPC virtual node listens to.


## Start Unity Simulator
1. Open Unity/Streaming project.
2. In Window -> ROS Settings, set the "Override Unity IP Address" to "host.docker.internal". This is needed so that the docker image can bind to the socket that Unity opens; existing implementation has issues with IPv6 addresses.
3. Press Play

## Frontend
```console
$ cd ./Frontend
$ npm start
```

## Streaming WebApp
```console
$ cd ./WebApp
$ npm start
```

