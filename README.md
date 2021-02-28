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
$ docker run -it --rm -p 10000:10000 -p 50051:50051 -p 60061:60061 robottycoon.azurecr.io/unity-robotics-pick-and-place:alpha ./src/start.sh
```

Port 10000 is the port that ROS listens to for communication with the simulator. Port 50051 is the port that the GRPC virtual node listens to. Port 60061 is a server that serves a tail of the console output.


## Start Unity Simulator
1. Open Unity/Streaming project.
2. In Window -> ROS Settings, set the "Override Unity IP Address" to "host.docker.internal". This is needed so that the docker image can bind to the socket that Unity opens; existing implementation has issues with IPv6 addresses.
3. Press Play

## Frontend
Note: If /editor is not working, remember that you have to run `git submodule update --init --recursive` to retrieve the content of the ace editor repository
```console
$ cd ./Frontend
$ npm start
```

## Streaming WebApp
Note: If /index.html is not working, remember to run `npm run-script build` to rebuild from the typescript files under WebApp/src
```console
$ cd ./WebApp
$ npm start
```

## Protos & GRPC
If changing the virtual node RPC service or protos under protos, you will need to update the generated code.

First you need the python & node protoc tools.
```console
$ pip install grpcio-tools
$ npm config set unsafe-perm true
$ npm install protoc-gen-grpc -g
$ npm install grpc-tools -g

```

Then you can run the code generation scripts:

```console
$ ./gen_protos.sh
```

Or on Windows in PowerShell:
```console
> .\gen_protos.ps1
```

