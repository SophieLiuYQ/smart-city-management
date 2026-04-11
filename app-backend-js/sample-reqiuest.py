import requests

url = "https://neat-coats-raise.loca.lt/describe"

# Sending an actual image file
with open("/home/isayah/code_projects/operation_get_cracked/smart-city-management/banana.webp", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json()["description"])