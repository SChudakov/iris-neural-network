import os
import googleapiclient.discovery

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"] = '/home/semen/drive/python/neural-network/schudakov-cloud-computing-d87d5081b78b.json'

instances = [
    [6.7, 3.1, 4.7, 1.5],
    [4.6, 3.1, 1.5, 0.2],
]

PROJECT_ID = 'schudakov-cloud-computing'
MODEL_NAME = 'IrisPredictor'
VERSION_NAME = 'v1'

service = googleapiclient.discovery.build('ml', 'v1')
name = 'projects/{}/models/{}/versions/{}'.format(PROJECT_ID, MODEL_NAME, VERSION_NAME)

response = service.projects().predict(
    name=name,
    body={'instances': instances}
).execute()

if 'error' in response:
    raise RuntimeError(response['error'])
else:
    print(response['predictions'])
