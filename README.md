# MLEngineeringEpamCourse (Task 5)


## Hello, there are two options how you can deploy it:
1) Option 1 - using Docker
2) Option 2 - using Kubernetes


## How to work with API
There are two services: fastapi-server on localhost:8090 and mlflow-back on localhost:5000.\
There are four requests:
1) train_model - select the model and train it, also add logs to mlflow
2) is_model_trained - check that you trained the model
3) predict - return inference for texts
4) healthcheck - just a healthcheck


## Option 1
### Create docker containers (Option 1)
If you want to run app with Docker, you can create docker containers
```
docker-compose create --build
```

### Push them to DockerHub (optional)
This part mostly for me, just wanted to highlight, that for kubernetes we use containers from Docker Hub and not locals
```
docker login -u "USER_NAME"
docker tag fastapi-server USER_NAME/fastapi-server
docker push USER_NAME/fastapi-server
docker tag mlflow-back USER_NAME/mlflow-back
docker push USER_NAME/mlflow-back
```

### Run with docker (Option 1)
If you want to start app with Docker (you can add -d or use two terminals)
```
docker run -p 5000:5000 --name="mlflow" --ip 172.17.0.1 --rm mlflow-back
docker run -p 8090:8090 --rm -e MLFLOW_IP="http://172.17.0.2:5000/" fastapi-server
```


## Option 2
### Install Kubernetes & Minicube (for Linux)
Here can be a problems, but there are no a good guide how to install kubernetes without porblems
```
snap install kubectl --classic
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

### Run with kubernetes (Option 2)
First of all let's start minikube and deploy mlflow-back container
```
minikube start
kubectl create deployment mlflow-back --image=maksimkrug/mlflow-back
```
Well, now you need to send IP for mlflow-back to fastapi-server container. It's possible to grab it automatically, but not too obvious how to send to the pod (there are no -e flag like in Docker). And only one way that I found it yo grab IP and replace the last row in deployment.yaml file (if needed)
```
# Get pods IP
kubectl get pods -o custom-columns=POD_IP:.status.podIPs
# Also you can use this command to extract IP
kubectl describe pods mlflow-back | grep IP:| awk '{print $2}' | head -1
# Then insert IP to deployment.yaml
kubectl apply -f deployment.yaml
kubectl port-forward deployment/fastapi-server 8090:8090
kubectl port-forward deployment/mlflow-back 5000:5000
```
