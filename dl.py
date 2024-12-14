import requests
from datetime import datetime, timedelta

# Define the headers
headers = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "accept-language": "en,zh-CN;q=0.9,zh;q=0.8,en-US;q=0.7",
    "cache-control": "no-cache",
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-requested-with": "XMLHttpRequest",
    "cookie": ("_hjSessionUser_774800=eyJpZCI6IjU3MjgxNGU0LTQ3ZGMtNTU3OS04NGNjLWRkMjNlMmIyNzlmNyIsImNyZWF0ZWQiOjE3MjU4MDI3MDU4MTIsImV4aXN0aW5nIjp0cnVlfQ==; "
               "f2=7149798d55a78fd77032398b72c33f6aae438ba179e1e456e3442fa6acfb7725cb88af52a6cbd2c4eeaf08da073ab42b0d67bbaa8e20a45881e3c74380e937fe; "
               "optimizelyEndUserId=oeu1733421490150r0.18791171452759547; _gcl_au=1.1.1579923834.1733421491; _ga=GA1.1.442208650.1733421491; "
               "_hjSession_774800=eyJpZCI6IjJiOWE4ZWJiLWZmMDItNGM4Yy05YmRkLTRlNTI3ODQ1NDVkZSIsImMiOjE3MzM0MjE0OTE3OTgsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MH0=; "
               "platsessionid__expires=1738605494898; platsessionid=34bde1e7-7aef-42bd-a271-02cd268de081.QgfEyqfzV4BNooFeX4wSY4xIUlZsYVV%2FyD551a2LTog; "
               "_ga_NVWC1BELMR=GS1.1.1733421491.1.1.1733421546.5.0.0; _uetsid=7e096e10b33211ef81b8cf9556ce69bb; _uetvid=9f9f54306de711ef9fd9d7dad2a7ed3c"),
    "Referer": "https://artofproblemsolving.com/",
    "Referrer-Policy": "origin"
}

# Define the URL
url = "https://artofproblemsolving.com/m/greedcontrol/ajax.php"

# Define the start and end dates
start_date = datetime(2024, 8, 22)
end_date = datetime(2024, 10, 1)

# Iterate through each date in the range
current_date = start_date
while current_date <= end_date:
    # Format the date as required
    formatted_date = current_date.strftime("%Y-%m-%d")
    
    # Define the payload
    payload = f"action=stats&date={formatted_date}"
    
    # Send the POST request
    response = requests.post(url, headers=headers, data=payload)
    
    # Print the result for each date
    with open(f"data/{formatted_date}.json", "w") as file:
        file.write(response.text)
    
    # Move to the next date
    current_date += timedelta(days=1)
