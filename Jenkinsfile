// CI/CD pipeline for the SMS spam classifier.
//
// NOTE on environment: this expects `docker` to be usable from *inside* the
// Jenkins container (so it can build/push images and control the app
// container on the host). Jenkins itself doesn't ship with the Docker CLI —
// this requires mounting the host's docker.sock into the Jenkins container
// and installing the docker CLI package inside it ("Docker-outside-of-Docker").
// We set that up as a separate step before this pipeline can run its
// build/deploy stages — see README.md "Jenkins Docker access" section.

pipeline {
    agent any

    environment {
        IMAGE_NAME = "REPLACE_WITH_DOCKERHUB_USERNAME/sms-spam-classifier"
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-credentials')
    }

    stages {
        stage('Install dependencies') {
            steps {
                sh 'python3 -m venv venv'
                sh './venv/bin/pip install --no-cache-dir -r requirements.txt'
            }
        }

        stage('Run tests') {
            steps {
                sh './venv/bin/pytest tests/ -v'
            }
        }

        stage('Fetch data') {
            steps {
                sh './venv/bin/python app/fetch_data.py'
            }
        }

        stage('Train model') {
            steps {
                sh './venv/bin/python app/train.py'
            }
        }

        stage('Evaluate model') {
            steps {
                // Non-zero exit here fails the build — a regressed model
                // never reaches the image-build stage.
                sh './venv/bin/python app/evaluate.py'
            }
        }

        stage('Build Docker image') {
            steps {
                sh "docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} -t ${IMAGE_NAME}:latest ."
            }
        }

        stage('Push to Docker Hub') {
            steps {
                sh "echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin"
                sh "docker push ${IMAGE_NAME}:${BUILD_NUMBER}"
                sh "docker push ${IMAGE_NAME}:latest"
            }
        }

        stage('Deploy') {
            steps {
                // Same host as Jenkins in this setup, so this is a local
                // restart rather than an SSH hop. Swap for an `ssh user@host`
                // wrapped command if Jenkins and the app ever live on
                // different machines.
                sh '''
                    docker rm -f sms-spam-app || true
                    docker run -d --name sms-spam-app -p 8000:8000 --restart unless-stopped ${IMAGE_NAME}:latest
                '''
            }
        }
    }

    post {
        always {
            sh 'docker logout || true'
        }
        failure {
            echo 'Pipeline failed — check the stage logs above for which gate blocked it.'
        }
    }
}
