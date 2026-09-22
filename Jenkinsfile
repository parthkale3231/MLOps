pipeline {

    agent any

    environment {
        AWS_REGION = 'ap-south-1'
        AWS_ACCOUNT_ID = '521024928341'
        ECR_REPOSITORY = 'customer-churn-api'
        IMAGE_TAG = "${BUILD_NUMBER}"
        EC2_PUBLIC_IP = '3.7.109.246'
        AWS_SHARED_CREDENTIALS_FILE = 'C:\\Users\\Parth\\.aws\\credentials'
        AWS_CONFIG_FILE = 'C:\\Users\\Parth\\.aws\\config'
        PATH = "C:\\Program Files\\Amazon\\AWSCLIV2;C:\\Windows\\System32\\OpenSSH;C:\\Users\\Parth\\AppData\\Local\\Programs\\DockerDesktop\\resources\\bin;C:\\Users\\Parth\\AppData\\Local\\Programs\\Python\\Python312;C:\\Users\\Parth\\AppData\\Local\\Programs\\Python\\Python312\\Scripts;${env.PATH}"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                bat '''
                    python --version
                    if not exist .venv python -m venv .venv
                    .venv\\Scripts\\pip.exe install --upgrade pip
                    .venv\\Scripts\\pip.exe install -r requirements.txt
                '''
            }
        }

        stage('DVC Pull') {
            steps {
                bat '''
                    .venv\\Scripts\\dvc.exe config remote.local_storage.url C:/dvc-storage
                    .venv\\Scripts\\dvc.exe pull
                    if not exist data\\customer_churn.csv if exist C:\\customer-churn-ml\\data\\customer_churn.csv copy C:\\customer-churn-ml\\data\\customer_churn.csv data\\customer_churn.csv
                    .venv\\Scripts\\python.exe -c "from src.config import DATA_PATH; assert DATA_PATH.exists(), f'Data missing: {DATA_PATH}'; print('Dataset confirmed:', DATA_PATH)"
                '''
            }
        }

        stage('Train Model') {
            steps {
                bat '''
                    .venv\\Scripts\\python.exe -m src.train
                '''
            }
        }

        stage('Evaluate Model') {
            steps {
                bat '''
                    set MIN_F1=0.55
                    .venv\\Scripts\\python.exe -m src.evaluate
                '''
            }
        }

        stage('Run Tests') {
            steps {
                bat '''
                    .venv\\Scripts\\pytest.exe -q
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                bat '''
                    docker build -f Dockerfile.api -t %ECR_REPOSITORY%:%IMAGE_TAG% -t %ECR_REPOSITORY%:latest .
                '''
            }
        }

        stage('Login to ECR') {
            steps {
                withCredentials([
                    [$class: 'AmazonWebServicesCredentialsBinding',
                     credentialsId: 'aws-mlops']
                ]) {
                    bat '''
                        aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com
                    '''
                }
            }
        }

        stage('Push Image to ECR') {
            steps {
                withCredentials([
                    [$class: 'AmazonWebServicesCredentialsBinding',
                     credentialsId: 'aws-mlops']
                ]) {
                    bat '''
                        docker tag %ECR_REPOSITORY%:%IMAGE_TAG% %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPOSITORY%:%IMAGE_TAG%
                        docker tag %ECR_REPOSITORY%:%IMAGE_TAG% %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPOSITORY%:latest

                        docker push %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPOSITORY%:%IMAGE_TAG%
                        docker push %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPOSITORY%:latest
                    '''
                }
            }
        }

        stage('Deploy to EC2') {
             steps {
                sshagent(['ec2-deploy-key']) {

                bat '''
                @echo off

                echo ========================================
                echo Deploying to EC2: %EC2_PUBLIC_IP%
                echo ========================================

                echo.
                echo Testing SSH connection...
                
                ssh -o StrictHostKeyChecking=no ubuntu@%EC2_PUBLIC_IP% "echo SSH connection successful"

                if %ERRORLEVEL% NEQ 0 (
                    echo ERROR: SSH connection failed
                    exit /b 1
                )

                echo.
                echo ========================================
                echo Logging in to ECR on EC2
                echo ========================================

                ssh -o StrictHostKeyChecking=no ubuntu@%EC2_PUBLIC_IP% "aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com"

                if %ERRORLEVEL% NEQ 0 (
                    echo ERROR: ECR login failed on EC2
                    exit /b 1
                )

                echo.
                echo ========================================
                echo Pulling Docker image
                echo ========================================

                ssh -o StrictHostKeyChecking=no ubuntu@%EC2_PUBLIC_IP% "docker pull %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPOSITORY%:%IMAGE_TAG%"

                if %ERRORLEVEL% NEQ 0 (
                    echo ERROR: Docker pull failed
                    exit /b 1
                )

                echo.
                echo ========================================
                echo Stopping old container
                echo ========================================

                ssh -o StrictHostKeyChecking=no ubuntu@%EC2_PUBLIC_IP% "docker stop customer-churn-api || true"

                echo.
                echo ========================================
                echo Removing old container
                echo ========================================

                ssh -o StrictHostKeyChecking=no ubuntu@%EC2_PUBLIC_IP% "docker rm customer-churn-api || true"

                echo.
                echo ========================================
                echo Starting new container
                echo ========================================

                ssh -o StrictHostKeyChecking=no ubuntu@%EC2_PUBLIC_IP% "docker run -d --name customer-churn-api --restart unless-stopped -p 8000:8000 %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPOSITORY%:%IMAGE_TAG%"

                if %ERRORLEVEL% NEQ 0 (
                    echo ERROR: Docker container failed to start
                    exit /b 1
                )

                echo.
                echo ========================================
                echo Deployment successful
                echo ========================================

                echo Application:
                echo http://%EC2_PUBLIC_IP%:8000
            '''
        }
    }
}

        stage('Deploy to EKS') {
            steps {

                withCredentials([
                    [$class: 'AmazonWebServicesCredentialsBinding',
                     credentialsId: 'aws-mlops']
                ]) {

                    bat '''
                        @echo off

                        echo ========================================
                        echo Deploying to Amazon EKS
                        echo ========================================

                        echo.
                        echo AWS Identity:
                        aws sts get-caller-identity

                        echo.
                        echo ========================================
                        echo Updating kubeconfig
                        echo ========================================

                        aws eks update-kubeconfig ^
                            --region %AWS_REGION% ^
                            --name customer-churn

                        if %ERRORLEVEL% NEQ 0 (
                            echo ERROR: Failed to update kubeconfig
                            exit /b 1
                        )

                        echo.
                        echo ========================================
                        echo Checking EKS cluster
                        echo ========================================

                        kubectl get nodes

                        if %ERRORLEVEL% NEQ 0 (
                            echo ERROR: Cannot access EKS cluster
                            exit /b 1
                        )

                        echo.
                        echo ========================================
                        echo Creating namespace
                        echo ========================================

                        kubectl apply -f k8s/namespace.yaml

                        if %ERRORLEVEL% NEQ 0 (
                            echo ERROR: Namespace deployment failed
                            exit /b 1
                        )

                        echo.
                        echo ========================================
                        echo Applying ConfigMap
                        echo ========================================

                        kubectl apply -f k8s/configmap.yaml

                        if %ERRORLEVEL% NEQ 0 (
                            echo ERROR: ConfigMap deployment failed
                            exit /b 1
                        )

                        echo.
                        echo ========================================
                        echo Applying Deployment
                        echo ========================================

                        kubectl apply -f k8s/deployment.yaml

                        if %ERRORLEVEL% NEQ 0 (
                            echo ERROR: Deployment creation failed
                            exit /b 1
                        )

                        echo.
                        echo ========================================
                        echo Applying Service
                        echo ========================================

                        kubectl apply -f k8s/service.yaml

                        if %ERRORLEVEL% NEQ 0 (
                            echo ERROR: Service deployment failed
                            exit /b 1
                        )

                        echo.
                        echo ========================================
                        echo Applying HPA
                        echo ========================================

                        kubectl apply -f k8s/hpa.yaml

                        if %ERRORLEVEL% NEQ 0 (
                            echo ERROR: HPA deployment failed
                            exit /b 1
                        )

                        echo.
                        echo ========================================
                        echo Updating Docker image
                        echo ========================================

                        kubectl set image deployment/customer-churn-api ^
                            customer-churn-api=%AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPOSITORY%:%IMAGE_TAG% ^
                            --namespace ml-prod

                        if %ERRORLEVEL% NEQ 0 (
                            echo ERROR: Failed to update Kubernetes image
                            exit /b 1
                        )

                        echo.
                        echo ========================================
                        echo Waiting for rollout
                        echo ========================================

                        kubectl rollout status deployment/customer-churn-api ^
                            --namespace ml-prod ^
                            --timeout=5m

                        if %ERRORLEVEL% NEQ 0 (
                            echo ERROR: Kubernetes rollout failed
                            exit /b 1
                        )

                        echo.
                        echo ========================================
                        echo Kubernetes deployment successful
                        echo ========================================

                        echo.
                        echo Pods:
                        kubectl get pods -n ml-prod

                        echo.
                        echo Services:
                        kubectl get svc -n ml-prod

                        echo.
                        echo HPA:
                        kubectl get hpa -n ml-prod

                        echo.
                        echo Deployment:
                        kubectl get deployment customer-churn-api -n ml-prod

                        echo.
                        echo ========================================
                        echo EKS DEPLOYMENT COMPLETE
                        echo ========================================
                    '''
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts allowEmptyArchive: true, artifacts: 'models/*.joblib,logs/*.log'
        }
        success {
            echo "CI/CD Pipeline succeeded! Deployed to EC2 (http://${env.EC2_PUBLIC_IP}:8000)"
        }
        failure {
            echo "CI/CD Pipeline failed. Review stage logs above."
        }
    }
}
