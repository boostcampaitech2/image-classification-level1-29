import requests

def send_message_to_slack(info):
    url = "https://hooks.slack.com/services/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"
    payload = {"text": "모델 학습 완료!! -> " + info}
    requests.post(url, json=payload)