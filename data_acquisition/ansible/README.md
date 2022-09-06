# Ansible

Automating the creation of Kubernetes clusters in various environments, the deployment of our software stack, as well as the execution and monitoring of experiments.

### General

Create a virtual environment and install the required packages (`requirements.txt`).

Install the necessary ansible collections with `ansible-galaxy install -r requirements.yml`.

Consider making changes to `ansible.cfg`, e.g., if you use another username or a specific ssh key. You will need to create a file `credentials.yml` with the following properties:

```yaml
# dockerhub credentials
dockerhub_username: NAME
dockerhub_token: TOKEN
```
This is necessary in order to pull the previously built operator image (in `data_acquisition/backend`, you will need to do that before), assuming that you are using dockerhub as image platform and are having a private image repository.

### AWS

You need to edit `inventory.aws.ec2.yml`, so that an access key and secret key is provided. Further, extend `credentials.yml` by:

```yaml
# AWS configuration
aws_access_key: ACCESS_KEY
aws_secret_key: SECRET_KEY
aws_region: REGION
aws_image_ami_id: IMAGE_ID
aws_keyname: SSH_USERNAME
```

### GCP

You need to edit `inventory.gcp.yml`, so that a valid project is referenced. Further, extend `credentials.yml` by:

```yaml
# GCP configuration
gcp_project: PROJECT_NAME
gcp_cred_kind: serviceaccount
gcp_cred_file: PATH_TO_YOUR_CREDENTIALS_FILE
gcp_zone: ZONE
gcp_region: REGION
gcp_image: IMAGE_ID
```
## Example

In order to bootstrap the infrastructure in GCP, first execute:
```
ansible-playbook infrastructure-gcp-bootstrap.yml -i inventory.gcp.yml -e @credentials.yml -e @scenario-gcp-ml.yml
```
This will bootstrap the infrastructure using the previously created credentials file and with respect to the gcp-ml scenario, which is the one we used for initial data gathering and evaluation of our approach. Consider having a look into the file in order to better understand the parametrization of this experiment.

Next, you will want to deploy the software stack:
```
ansible-playbook experiment-bootstrap.yml -i inventory.gcp.yml -e @credentials.yml -e @scenario-gcp-ml.yml
```
This will deploy e.g. prometheus, chaos-mesh, kubestone, and our operator.

Finally, to start the respective experiment, issue the following command:
```
ansible-playbook experiment-run.yml -i inventory.gcp.yml -e @credentials.yml -e @scenario-gcp-ml.yml
```
Ansible is configured to wait for a certain period of time and periodically check if the experiment already finished, in order to fetch the data from the respective container.