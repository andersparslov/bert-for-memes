import requests
URL = "https://meme-review-xgikjkwjna-ew.a.run.app"
LABEL_DESCRIPTION = {
    0 : "Not funny",
    1 : "Funny",
    2 : "Very funny",
    3 : "Hilarious"
}

def request(meme):
    response = requests.get(URL, {"meme":meme}).json()['response']
    print(LABEL_DESCRIPTION[response])


if __name__ == '__main__':
    meme = input("Enter your meme here: ")
    request(meme)