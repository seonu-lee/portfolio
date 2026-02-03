import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
import os
import re

# 간단한 설정
CATEGORIES = {
    'business': 'https://edition.cnn.com/business',
    'markets': 'https://edition.cnn.com/business/markets',
    'entertainment': 'https://edition.cnn.com/entertainment',
    'tech': 'https://edition.cnn.com/business/tech'
}

ARTICLES_PER_CATEGORY = 25
OUTPUT_DIR = "data"

class CNNScraper:
    def __init__(self):
        self.articles = []
        self.setup_driver()
    
    def setup_driver(self):
        """Selenium 드라이버 설정 - 수동 ChromeDriver"""
        options = Options()
        
        # Chrome 옵션 설정
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # GPU 에러 무시
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--log-level=3')  # 에러 메시지만 표시
        
        try:
            # 방법 1: 현재 폴더의 chromedriver.exe 사용
            if os.path.exists('chromedriver.exe'):
                print(" 로컬 chromedriver.exe 발견")
                service = Service('./chromedriver.exe')
                self.driver = webdriver.Chrome(service=service, options=options)
                print(" Chrome 드라이버 초기화 완료 (로컬)")
            else:
                # 방법 2: 시스템 PATH의 chromedriver 사용
                print(" 시스템 ChromeDriver 사용 시도...")
                self.driver = webdriver.Chrome(options=options)
                print(" Chrome 드라이버 초기화 완료 (시스템)")
                
        except Exception as e:
            print(f"\n ChromeDriver 초기화 실패: {e}")
            print("\n" + "="*60)
            print(" 해결 방법:")
            print("="*60)
            print("\n1. ChromeDriver 다운로드:")
            print("   https://googlechromelabs.github.io/chrome-for-testing/")
            print("\n2. Chrome 144 버전용 다운로드:")
            print("   - 'chrome-win64.zip' 또는 'chromedriver-win64.zip' 검색")
            print("   - 버전: 144.0.7559.110 또는 144.0.7559.x")
            print("\n3. 다운로드 후:")
            print("   - chromedriver.exe 압축 해제")
            print("   - 이 스크립트와 같은 폴더에 복사")
            print(f"   - 현재 폴더: {os.getcwd()}")
            print("\n또는 명령어로 자동 다운로드:")
            print("   python download_chromedriver.py")
            print("="*60)
            raise e
    
    def scrape_category(self, category_name, category_url):
        """카테고리별 기사 수집"""
        print(f"\n {category_name.upper()} 카테고리 수집 중...")
        
        try:
            # 1. 카테고리 페이지 열기
            self.driver.get(category_url)
            print(f"   페이지 로딩 중: {category_url}")
            time.sleep(5)
            
            # 2. 스크롤 내려서 더 많은 기사 로드
            self.scroll_to_load_more()
            
            # 3. 기사 링크 추출
            article_links = self.extract_article_links()
            print(f"   발견된 기사 링크: {len(article_links)}개")
            
            if not article_links:
                print(f"    {category_name}에서 기사를 찾지 못했습니다.")
                return
            
            # 디버깅: 처음 3개 링크 출력
            print(f"   예시 링크:")
            for link in article_links[:3]:
                print(f"      - {link}")
            
            # 4. 각 기사 상세 정보 수집
            collected = 0
            for link in tqdm(article_links[:ARTICLES_PER_CATEGORY], desc=f"   {category_name}"):
                article_data = self.scrape_article(link, category_name)
                if article_data:
                    self.articles.append(article_data)
                    collected += 1
                time.sleep(2)
            
            print(f"    {collected}개 기사 수집 완료")
            
        except Exception as e:
            print(f"    {category_name} 카테고리 수집 중 에러: {e}")
    
    def scroll_to_load_more(self):
        """페이지 스크롤하여 동적 콘텐츠 로드"""
        SCROLL_PAUSE_TIME = 2
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        print("    페이지 스크롤 중...")
        for i in range(5):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        print("    스크롤 완료")
    
    def extract_article_links(self):
        """기사 링크 추출 - CNN 패턴"""
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        links = []
        seen_urls = set()
        
        # CNN 기사 URL 패턴: /YYYY/MM/DD/category/article-title
        cnn_article_pattern = re.compile(r'/\d{4}/\d{2}/\d{2}/[^/]+/[^/]+')
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            if cnn_article_pattern.search(href):
                if href.startswith('/'):
                    full_url = 'https://edition.cnn.com' + href
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue
                
                full_url = full_url.replace('www.cnn.com', 'edition.cnn.com')
                full_url = full_url.split('?')[0].split('#')[0]
                
                if full_url not in seen_urls:
                    if not any(x in full_url for x in ['/videos/', '/gallery/', '/live-news/', '/video/']):
                        links.append(full_url)
                        seen_urls.add(full_url)
        
        return links
    
    def scrape_article(self, url, category):
        """개별 기사 상세 정보 수집"""
        try:
            self.driver.get(url)
            time.sleep(3)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # 제목
            title = "N/A"
            for selector in ['h1[id="maincontent"]', 'h1.headline__text', 'h1', '[data-editable="headline"]']:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    if title and len(title) > 5:
                        break
            
            # 본문
            content = ""
            paragraphs = soup.select('div.article__content p.paragraph')
            
            if not paragraphs:
                article_body = soup.find('div', class_=lambda x: x and 'article__content' in str(x))
                if article_body:
                    paragraphs = article_body.find_all('p')
            
            if not paragraphs:
                paragraphs = soup.find_all('p')
            
            if paragraphs:
                valid_paragraphs = [
                    p.get_text(strip=True) 
                    for p in paragraphs 
                    if len(p.get_text(strip=True)) > 50
                ]
                content = ' '.join(valid_paragraphs)
            
            if not content or len(content) < 100:
                content = "본문을 찾을 수 없습니다."
            
            # 작성일
            published_date = "N/A"
            for meta_property in ['article:published_time', 'pubdate', 'datePublished']:
                date_tag = soup.find('meta', property=meta_property) or soup.find('meta', {'name': meta_property})
                if date_tag and date_tag.get('content'):
                    published_date = date_tag['content']
                    break
            
            if published_date == "N/A":
                time_tag = soup.find('time')
                if time_tag:
                    published_date = time_tag.get('datetime', time_tag.get_text(strip=True))
            
            # 저자
            author = "N/A"
            author_meta = soup.find('meta', property='article:author') or soup.find('meta', {'name': 'author'})
            if author_meta and author_meta.get('content'):
                author = author_meta['content']
            
            if author == "N/A":
                byline = soup.find(class_=lambda x: x and 'byline' in str(x).lower())
                if byline:
                    author = byline.get_text(strip=True)
            
            # 이미지
            image_url = "N/A"
            img_meta = soup.find('meta', property='og:image')
            if img_meta and img_meta.get('content'):
                image_url = img_meta['content']
            
            # 태그
            tags = []
            keywords_meta = soup.find('meta', {'name': 'keywords'})
            if keywords_meta and keywords_meta.get('content'):
                tags = [k.strip() for k in keywords_meta['content'].split(',')]
            
            tag_links = soup.find_all('a', class_=lambda x: x and 'tag' in str(x).lower())
            for tag_link in tag_links:
                tag_text = tag_link.get_text(strip=True)
                if tag_text and tag_text not in tags:
                    tags.append(tag_text)
            
            return {
                'title': title,
                'content': content[:10000],
                'published_date': published_date,
                'url': url,
                'author': author,
                'category': category,
                'tags': '|'.join(tags[:10]) if tags else "N/A",
                'image_url': image_url,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"\n       기사 수집 실패: {str(e)[:50]}")
            return None
    
    def save_to_csv(self):
        """CSV로 저장"""
        if not self.articles:
            print("\n 수집된 기사가 없습니다.")
            return
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = pd.DataFrame(self.articles)
        df = df.drop_duplicates(subset=['url'], keep='first')
        
        print(f"\n CSV 파일 저장 중...")
        print(f"   총 {len(df)}개 기사 (중복 제거 후)")
        
        for category in CATEGORIES.keys():
            category_df = df[df['category'] == category]
            if not category_df.empty:
                filename = f"{OUTPUT_DIR}/{category}_articles.csv"
                category_df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"    {category.upper()}: {len(category_df)}개 → {filename}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        all_filename = f"{OUTPUT_DIR}/all_articles_{timestamp}.csv"
        df.to_csv(all_filename, index=False, encoding='utf-8-sig')
        print(f"\n    전체: {len(df)}개 → {all_filename}")
        
        print(f"\n 수집 통계:")
        print(df['category'].value_counts().to_string())
    
    def run(self):
        """전체 실행"""
        print("=" * 60)
        print(" CNN 뉴스 수집 파이프라인 시작!")
        print("=" * 60)
        
        start_time = time.time()
        
        for category_name, category_url in CATEGORIES.items():
            self.scrape_category(category_name, category_url)
        
        self.save_to_csv()
        
        try:
            self.driver.quit()
        except:
            pass
        
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f" 수집 완료! (소요 시간: {elapsed_time/60:.2f}분)")
        print("=" * 60)

if __name__ == "__main__":
    try:
        scraper = CNNScraper()
        scraper.run()
    except KeyboardInterrupt:
        print("\n\n 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n 예상치 못한 에러 발생: {e}")