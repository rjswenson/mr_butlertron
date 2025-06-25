mport requests
import json

url = 'http://localhost:5000/todo/api/v1.0/tasks'

response = requests.get(url)

print(str(response))
print('')
print(json.dumps(response.json(), indent=4))