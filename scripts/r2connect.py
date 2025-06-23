import requests

urls = [
    "https://pub-e12d369657534f328cc36a7331ff7bff.r2.dev/wikipedia_vectors_manifest.json",
    "https://pub-e12d369657534f328cc36a7331ff7bff.r2.dev/wikipedia_vectors_faiss.index",
    "https://pub-e12d369657534f328cc36a7331ff7bff.r2.dev/wikipedia_vectors.mp4"
]

for url in urls:
    try:
        response = requests.head(url, timeout=10)
        size_mb = int(response.headers.get('content-length', 0)) / 1024 / 1024
        print(f"✅ {url.split('/')[-1]}: {response.status_code} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ {url.split('/')[-1]}: {e}")