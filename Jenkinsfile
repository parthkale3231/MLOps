pipeline {

    agent any

    environment {
        AWS_REGION = 'ap-south-1'
        AWS_ACCOUNT_ID = '521024928341'
        ECR_REPOSITORY = 'customer-churn-api'
        IMAGE_TAG = "${BUILD_NUMBER}"
        AWS_SHARED_CREDENTIALS_FILE = 'C:\\Users\\Parth\\.aws\\credentials'
        AWS_CONFIG_FILE = 'C:\\Users\\Parth\\.aws\\config'
        PATH = "C:\\Program Files\\Amazon\\AWSCLIV2;C:\\Users\\Parth\\AppData\\Local\\Programs\\DockerDesktop\\resources\\bin;C:\\Users\\Parth\\AppData\\Local\\Programs\\Python\\Python312;C:\\Users\\Parth\\AppData\\Local\\Programs\\Python\\Python312\\Scripts;${env.PATH}"
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
    }

    post {
        always {
            archiveArtifacts allowEmptyArchive: true, artifacts: 'models/*.joblib,logs/*.log'
        }
        success {
            echo "CI/CD Pipeline succeeded! Image pushed to ECR: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
        }
        failure {
            echo "CI/CD Pipeline failed. Review stage logs above."
        }
    }
}
