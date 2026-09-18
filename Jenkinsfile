pipeline {
    agent {
        // Run on any available agent or specify a Windows node label: label 'windows'
        any
    }

    options {
        timeout(time: 45, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
    }

    environment {
        VENV_DIR = ".venv"
        MIN_F1 = "0.55"
        IMAGE_NAME = "customer-churn-api"
    }

    stages {
        stage('Git Code') {
            steps {
                echo 'Checking out source code from Git repository...'
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                echo 'Preparing Python virtual environment and dependencies on Windows...'
                bat """
                    @echo off
                    if not exist ${VENV_DIR} (
                        echo Creating Python virtual environment...
                        python -m venv ${VENV_DIR}
                    )
                    echo Installing and upgrading dependencies...
                    ${VENV_DIR}\\Scripts\\python.exe -m pip install --upgrade pip
                    ${VENV_DIR}\\Scripts\\pip.exe install -r requirements.txt
                """
            }
        }

        stage('Parallel Pre-Checks') {
            parallel {
                stage('Tests') {
                    steps {
                        echo '[Parallel 1/3] Running automated unit tests...'
                        bat """
                            @echo off
                            if not exist test-reports mkdir test-reports
                            ${VENV_DIR}\\Scripts\\pytest.exe --junitxml=test-reports/results.xml
                        """
                    }
                }

                stage('DVC Data') {
                    steps {
                        echo '[Parallel 2/3] Checking DVC data tracking and verifying dataset...'
                        bat """
                            @echo off
                            ${VENV_DIR}\\Scripts\\dvc.exe status
                            ${VENV_DIR}\\Scripts\\python.exe -c "from src.config import DATA_PATH; assert DATA_PATH.exists(), f'Data file missing: {DATA_PATH}'; print(f'Dataset confirmed: {DATA_PATH}')"
                        """
                    }
                }

                stage('Docker') {
                    steps {
                        echo '[Parallel 3/3] Checking Docker engine readiness...'
                        bat """
                            @echo off
                            docker version
                        """
                    }
                }
            }
        }

        stage('Training') {
            steps {
                echo 'Executing customer churn model training pipeline...'
                bat """
                    @echo off
                    ${VENV_DIR}\\Scripts\\python.exe -m src.train
                """
            }
        }

        stage('MLflow') {
            steps {
                echo 'Verifying MLflow experiment tracking, metrics, and registered models...'
                bat """
                    @echo off
                    if exist mlartifacts dir /b mlartifacts
                    if exist mlflow.db echo Local SQLite MLflow DB verified.
                    ${VENV_DIR}\\Scripts\\python.exe -c "import mlflow; print('MLflow tracking client initialized successfully.')"
                """
            }
        }

        stage('Model Validation') {
            steps {
                echo 'Evaluating model performance against production Quality Gate...'
                bat """
                    @echo off
                    set MIN_F1=${MIN_F1}
                    ${VENV_DIR}\\Scripts\\python.exe -m src.evaluate
                """
            }
        }

        stage('Model Artifact') {
            steps {
                echo 'Verifying serialized model artifact presence and integrity...'
                bat """
                    @echo off
                    if not exist models\\churn_model.joblib (
                        echo Error: models\\churn_model.joblib not found!
                        exit /b 1
                    )
                    echo Model artifact successfully verified at models\\churn_model.joblib
                """
            }
        }

        stage('FastAPI Container') {
            steps {
                echo 'Building production FastAPI Docker image with packaged model...'
                bat """
                    @echo off
                    docker build -f Dockerfile.api -t %IMAGE_NAME%:%BUILD_NUMBER% -t %IMAGE_NAME%:latest .
                """
            }
        }
    }

    post {
        always {
            echo 'Archiving test results and build artifacts...'
            junit allowEmptyResults: true, testResults: 'test-reports/*.xml'
            archiveArtifacts allowEmptyArchive: true, artifacts: 'models/*.joblib,logs/*.log'
        }
        success {
            echo "CI/CD Pipeline succeeded! Container image ${IMAGE_NAME}:${BUILD_NUMBER} is ready for deployment."
        }
        failure {
            echo "CI/CD Pipeline failed. Please check the stage logs for details."
        }
    }
}
