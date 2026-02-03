# CNN API 설정
CNN_API_KEY = "c0556b987dmshf63d9e426d5240dp1a4b25jsne469529d596d"  # API 키
CNN_API_BASE_URL = "cnnapi.p.rapidapi.com"  # 예시 URL

# 카테고리 설정
CATEGORIES = {
    'business': 'https://www.cnn.com/business',
    'markets': 'https://www.cnn.com/markets',
    'entertainment': 'https://www.cnn.com/entertainment',
    'tech': 'https://www.cnn.com/tech'
}

# 수집 설정
ARTICLES_PER_CATEGORY = 25  # 카테고리당 25개 = 총 100개
DAYS_TO_SCRAPE = 30  # 최근 1개월

# CSV 저장 경로
OUTPUT_DIR = "data"