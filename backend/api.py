import requests

url = "https://api.genoox.com/v1/auth/login?email=robin2004312@st.jmi.ac.in"

payload={}
headers = {
  'Authorization': 'Rob@04051998'
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
