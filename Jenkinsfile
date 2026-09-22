pipeline {
    agent any

    environment {
        MIN_F1 = '0.55'
        IMAGE_NAME = 'customer-churn-api'
        PATH = "C:\\Users\\Parth\\AppData\\Local\\Programs\\Python\\Python312;C:\\Users\\Parth\\AppData\\Local\\Programs\\Python\\Python312\\Scripts;C:\\Users\\Parth\\AppData\\Local\\Programs\\DockerDesktop\\resources\\bin;${env.PATH}"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python') {
            steps {
                bat '''
                    python --version
                    if not exist .venv python -m venv .venv
                    .venv\\Scripts\\pip.exe install --upgrade pip
                    .venv\\Scripts\\pip.exe install -r requirements.txt
                '''
            }
        }

        stage('Pre-Checks') {
            parallel {
                stage('Tests') {
                    steps {
                        bat '''
                            .venv\\Scripts\\pytest.exe
                        '''
                    }
                }

                stage('DVC Data') {
                    steps {
                        bat '''
                            .venv\\Scripts\\dvc.exe status
                            .venv\\Scripts\\python.exe -c "from src.config import DATA_PATH; assert DATA_PATH.exists(), f'Data missing: {DATA_PATH}'; print('Dataset confirmed:', DATA_PATH)"
                        '''
                    }
                }

                stage('Docker') {
                    steps {
                        bat '''
                            docker version
                        '''
                    }
                }
            }
        }

        stage('Train Model') {
            steps {
                bat '''
                    .venv\\Scripts\\python.exe src/train.py
                '''
            }
        }

        stage('MLflow') {
            steps {
                bat '''
                    .venv\\Scripts\\python.exe -c "import mlflow; print('MLflow tracking verified')"
                    if exist mlartifacts dir mlartifacts
                '''
            }
        }

        stage('Model Validation') {
            steps {
                bat '''
                    set MIN_F1=0.55
                    .venv\\Scripts\\python.exe src/evaluate.py
                '''
            }
        }

        stage('Model Artifact') {
            steps {
                bat '''
                    if not exist models\\churn_model.joblib exit /b 1
                    echo Model artifact verified: models\\churn_model.joblib
                '''
            }
        }

        stage('FastAPI Container') {
            steps {
                bat """
                    docker build -f Dockerfile.api -t %IMAGE_NAME%:%BUILD_NUMBER% -t %IMAGE_NAME%:latest .
                """
            }
        }
    }

    post {
        always {
            archiveArtifacts allowEmptyArchive: true, artifacts: 'models/*.joblib,logs/*.log'
        }
        success {
            echo "Pipeline succeeded! Container image ${IMAGE_NAME}:${BUILD_NUMBER} is ready."
        }
        failure {
            echo "Pipeline failed. Review stage logs above."
        }
    }
}
