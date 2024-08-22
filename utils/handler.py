import requests
import json
def handle_response(response):
    try:
        response.raise_for_status()  # HTTP 오류가 있는 경우 예외 발생
        return response.json()
    except json.decoder.JSONDecodeError as json_err:
        print(response.status_code)
        print(response.json())
        return {"error": "JSON decode error occurred", "message": str(json_err)}
    except requests.exceptions.RequestException as req_err:
        print(response.status_code)
        print(response.json())
        return {"error": "Request error occurred", "message": str(req_err)}
    except requests.exceptions.HTTPError as http_err:
        print(response.status_code)
        print(response.json())
        return {"error": "HTTP error occurred", "message": str(http_err)}


