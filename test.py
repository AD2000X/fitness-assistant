import requests

def test_question_api():
    try:
        url = "http://localhost:5000/question"
        test_question = (
            "Is the Lat Pulldown considered a strengh training activity, and if so, why?"
            )
        data = {"question": test_question} 
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.json().get("conversation_id")
    except Exception as e:
        print(f"Error testing question API: {e}")
        return None

def test_feedback_api(conv_id):
    try:
        url = "http://localhost:5000/feedback"
        data = {
            "conversation_id": conv_id,
            "feedback": 1
        }
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error testing feedback API: {e}")

def main():
    print("Starting API tests...")
    conv_id = test_question_api()
    if conv_id:
        test_feedback_api(conv_id)
    print("Tests completed!")

if __name__ == "__main__":
    main()

# # simple test in terminal step_1:
# URL=http://localhost:5000

# DATA='{
#     "question": "Is the Lat Pulldown considered a strengh training activity, and if so, why?"
# }'

# curl -X POST \
#   -H "Content-Type: application/json" \
#   -d "${DATA}" \
#   ${URL}/question


# # simple test in terminal step_2:
# ID="77cb4248-1abb-49c0-8049-2c5cf9f8eb65"
# URL="http://127.0.0.1:5000"

# FEEDBACK_DATA='{
#     "conversation_id": "'${ID}'",
#     "feedback": 1
# }'

# curl -X POST \
#   -H "Content-Type: application/json" \
#   -d "${FEEDBACK_DATA}" \
#   ${URL}/feedback


# # simple test in terminal step_3:
# import requests

# url = "http://localhost:5000/question"

# question = (
#     "Is the Lat Pulldown considered a strengh training activity, and if so, why?"
# )
# data = {"question": question}

# response = requests.post(url, json=data)
# # print(response.content)

# print(response.json())