# ☁️ Cloud Deployment Guide: Docker & Kubernetes (K8s)

This guide provides the exact commands to containerize **Phishing Shield AI** and deploy it to any cloud Kubernetes cluster (AWS EKS, Google Cloud GKE, Azure AKS, or local Minikube / Docker Desktop Kubernetes).

---

## 📦 Step 1: Build & Push the Docker Image

### 1.1 Ensure Docker Desktop is Running
Make sure Docker Desktop is started on your computer.

### 1.2 Login to Docker Hub (or Cloud Container Registry)
```bash
docker login
```

### 1.3 Build the Docker Image
Replace `yourusername` with your Docker Hub username:
```bash
docker build -t yourusername/phishing-shield:latest .
```

### 1.4 Test the Container Locally
```bash
docker run -p 5000:5000 yourusername/phishing-shield:latest
```
Visit `http://localhost:5000` to verify it works, then press `Ctrl + C`.

### 1.5 Push to Docker Hub / Cloud Registry
```bash
docker push yourusername/phishing-shield:latest
```

---

## ☸️ Step 2: Deploy to Kubernetes

### 2.1 Update the Image Name in `k8s/deployment.yaml`
Open `k8s/deployment.yaml` and replace:
```yaml
image: <YOUR_DOCKERHUB_OR_REGISTRY_USER>/phishing-shield:latest
```
with your actual image name, e.g.:
```yaml
image: yourusername/phishing-shield:latest
```

### 2.2 Apply the Deployment & Service
Run:
```bash
kubectl apply -f k8s/deployment.yaml
```

### 2.3 Verify the Pods & Services
```bash
# Check running pods
kubectl get pods -l app=phishing-shield

# Check the LoadBalancer service
kubectl get services phishing-shield-service
```

Once the `EXTERNAL-IP` changes from `<pending>` to an IP address (or AWS load balancer DNS name), you can visit:
```
http://<EXTERNAL-IP>
```

---

## 🌐 Cloud Specific Quick References

### Google Cloud (GKE)
```bash
# 1. Connect to your GKE cluster
gcloud container clusters get-credentials my-cluster --zone us-central1-a

# 2. Deploy
kubectl apply -f k8s/deployment.yaml
```

### AWS (EKS)
```bash
# 1. Connect to your EKS cluster
aws eks update-kubeconfig --region us-east-1 --name my-cluster

# 2. Deploy
kubectl apply -f k8s/deployment.yaml
```

### Azure (AKS)
```bash
# 1. Connect to your AKS cluster
az aks get-credentials --resource-group myRG --name myAKSCluster

# 2. Deploy
kubectl apply -f k8s/deployment.yaml
```

---

## 🛑 How to Delete the Deployment
```bash
kubectl delete -f k8s/deployment.yaml
```

