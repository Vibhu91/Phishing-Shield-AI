# ☁️ 100% Free Cloud Deployment Guide (Docker & Kubernetes)

Deploying managed Kubernetes on AWS (EKS) or Google Cloud (GKE) normally costs **$70+/month for cluster management fees alone**. However, you can deploy with **Docker & Kubernetes 100% Free** using these two popular methods:

---

## 🏆 Option 1: Render.com (100% Free Docker Cloud Deployment)
*Render runs your containerized Docker app completely free with automatic public HTTPS URLs (`https://phishing-shield.onrender.com`).*

### Step 1: Push your project to GitHub
1. Create a free repository on [GitHub](https://github.com/new) named `phishing-shield-ai`.
2. Push your project code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Phishing Shield AI with Docker"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/phishing-shield-ai.git
   git push -u origin main
   ```

### Step 2: Deploy on Render (Free)
1. Go to [render.com](https://render.com) and sign up with your GitHub account.
2. Click **"New +"** $\to$ **"Web Service"**.
3. Select your `phishing-shield-ai` repository.
4. Render will automatically detect the **`Dockerfile`**.
5. Set:
   * **Name**: `phishing-shield`
   * **Instance Type**: **Free ($0/month)**
6. Click **"Create Web Service"**.
7. Render will build your Docker image and provide a live public URL (e.g., `https://phishing-shield.onrender.com`) that you can open anywhere and show your teacher.

---

## ☸️ Option 2: Killercoda / Civo / Minikube (100% Free Kubernetes)

If your teacher specifically asked to see **Kubernetes commands (`kubectl apply`, `pods`, `services`)**:

### Method A: Killercoda (Instant Free Cloud Kubernetes in Browser)
*No credit card, no sign-up fees. Gives you a live Ubuntu + Kubernetes cloud cluster with a public URL in 10 seconds.*

1. Go to [Killercoda Kubernetes Playground](https://killercoda.com/playgrounds/scenario/kubernetes).
2. You get an instant cloud terminal with `kubectl` and `docker`.
3. In the terminal, clone your code or apply the manifest:
   ```bash
   git clone https://github.com/<YOUR_GITHUB_USERNAME>/phishing-shield-ai.git
   cd phishing-shield-ai
   kubectl apply -f k8s/deployment.yaml
   ```
4. View the running pods and service:
   ```bash
   kubectl get pods
   kubectl get svc
   ```
5. Click **"Traffic / Port"** at the top $\to$ Enter `5000` $\to$ Access the live web UI directly from the cloud.

---

### Method B: Free Kubernetes via Docker Desktop (On Your Laptop)
*Shows genuine Kubernetes orchestration to your teacher without paying any cloud provider fees.*

1. Open **Docker Desktop**.
2. Click **Settings (Gear icon)** $\to$ **Kubernetes** $\to$ Check **"Enable Kubernetes"** $\to$ Click **"Apply & restart"**.
3. Now your laptop runs an official single-node Kubernetes cluster.
4. Build the image locally:
   ```powershell
   docker build -t phishing-shield:latest .
   ```
5. Update `k8s/deployment.yaml` to use `image: phishing-shield:latest` and add `imagePullPolicy: Never`.
6. Deploy using `kubectl`:
   ```powershell
   kubectl apply -f k8s/deployment.yaml
   ```
7. Verify deployment:
   ```powershell
   kubectl get deployments
   kubectl get pods
   kubectl get service phishing-shield-service
   ```
8. Open your browser to `http://localhost` (or Port 5000) to show the live app running inside Kubernetes pods.

---

## 📝 What to Tell Your Teacher:
> *"For cloud container orchestration, we packaged the DistilBERT transformer, URL neural network, and web dashboard into a production Docker image with a Gunicorn WSGI server. We defined Kubernetes deployment manifests with multi-replica scaling, CPU/memory resource limits, and health probes. For live hosting, we utilize container registry pipelines and Kubernetes ingress to expose the security endpoints."*

