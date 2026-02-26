import nltk

resources = ['all']

def download_nltk_resources():
    nltk.download(resources,quiet=True)

if __name__ == "__main__":
    download_nltk_resources()
    print("NLTK resources downloaded successfully.")