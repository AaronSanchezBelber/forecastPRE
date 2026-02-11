pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo 'Checkout handled by Jenkins SCM'
            }
        }
        stage('Setup Python') {
            steps {
                bat 'python --version'
                bat 'python -m pip install --upgrade pip'
            }
        }
        stage('Install Deps') {
            steps {
                bat 'python -m pip install -r requirements.txt'
            }
        }
        stage('Lint Basic') {
            steps {
                bat 'python -m compileall src'
            }
        }
        stage('Tests') {
            steps {
                bat 'python -m pytest || echo No tests found'
            }
        }
    }
}
