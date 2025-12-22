--1341. Movie Rating
/*문제: Find the name of the user who has rated the greatest number of movies. In case of a tie, return the lexicographically smaller user name. (가장 많은 영화를 평점한 사용자의 이름을 찾아라, 동점일 경우 사전적으로 작은 사용자의 이름을 찾아라)
Find the movie name with the highest average rating in February 2020. In case of a tie, return the lexicographically smaller movie name. (2020년 2월에 평균 평점이 가장 높은 영화의 제목을 찾아라. 동점일 경우 사전적으로 작은 영화의 이름을 반환) */

-- 첫 번째 SELECT: 리뷰를 가장 많이 한 사용자의 이름을 조회
(SELECT name AS results
FROM MovieRating JOIN Users USING(user_id)
GROUP BY name  -- 사용자별로 그룹화 (리뷰 개수를 세기 위함)
ORDER BY COUNT(*) DESC, name   -- 리뷰 개수가 많은 순서대로 내림차순 정렬, 동률이면 이름 오름차순
LIMIT 1)  -- 가장 리뷰를 많이 한 사용자 1명만 선택

UNION ALL -- 두 SELECT 결과를 합치되, 중복도 허용 (즉, 단순 연결)

-- 두 번째 SELECT: 2020년 2월에 평균 평점이 가장 높은 영화 제목을 조회
(SELECT title AS results
FROM MovieRating JOIN Movies USING(movie_id)
WHERE EXTRACT(YEAR_MONTH FROM created_at) = 202002  -- 리뷰 작성일이 2020년 2월인 경우만 선택
GROUP BY title  -- 영화 제목별로 그룹화 (평균 평점 계산을 위해)
ORDER BY AVG(rating) DESC, title   -- 평균 평점이 높은 순서대로 내림차순 정렬, 동률이면 제목 오름차순
LIMIT 1); -- 평균 평점이 가장 높은 영화 1편만 선택

/* EXTRACT(단위 FROM 날짜컬럼) :날짜에서 특정 부분(연도, 월, 일, 시 등)을 추출할 때 사용
단위에는 YEAR, MONTH, DAY, HOUR, YEAR_MONTH 등 다양한 값이 올 수 있습니다.
날짜컬럼은 DATE, DATETIME, TIMESTAMP 형식의 컬럼입니다. */

--1393. Capital Gain/Loss
/*문제: Write a solution to report the Capital gain/loss for each stock. (각 주식에 대한 자본 이득/손실을 보고해라)
The Capital gain/loss of a stock is the total gain or loss after buying and selling the stock one or many times. (주식의 자본 이득/손실은 주식을 한 번 또는 여러번 사고 팔았을 때의 총 이득 또는 손실을 뜻함)
Return the result table in any order. */

Select stock_name, SUM(CASE when operation = 'Sell' then price else -price end) as capital_gain_loss
from Stocks 
group by stock_name;

--1907. Count Salary Categories
/*문제: Write a solution to calculate the number of bank accounts for each salary category. The salary categories are: (각 급여 카테고리에 대한 은행 계좌수를 계산해라)
"Low Salary": All the salaries strictly less than $20000.
"Average Salary": All the salaries in the inclusive range [$20000, $50000].
"High Salary": All the salaries strictly greater than $50000.
(낮은급여: 2만달러미만, 평균급여: 2만달러~5만달러, 높은급여: 5만달러 초과)
The result table must contain all three categories. If there are no accounts in a category, return 0. (3가지카테고리 모두 포함, 카테고리에 없는 계정은 0을 반환)
Return the result table in any order. */

(SELECT
    'Low Salary' As category
    ,COUNT(CASE WHEN income<20000 THEN 1 END) AS accounts_count
FROM accounts)

UNION ALL

(SELECT
    'Average Salary' AS category
    ,COUNT(CASE WHEN income BETWEEN 20000 AND 50000 THEN 1 END) AS accounts_count
FROM accounts)

UNION ALL

(SELECT 
    'High Salary' AS category
    ,COUNT(CASE WHEN income>50000 THEN 1 END) AS accounts_count
FROM accounts)


--1934. Confirmation Rate
/*문제: The confirmation rate of a user is the number of 'confirmed' messages divided by the total number of requested confirmation messages. The confirmation rate of a user that did not request any confirmation messages is 0. Round the confirmation rate to two decimal places. ( 사용자의 확인율은 확인된 메세지의 수를 요청된 확인 메세지 수로 나눈 값임, 확인메세지를 요청하지 않은 사용자의 확인율은 0임. 확인율을 소수점 두자리로 반올림)
Write a solution to find the confirmation rate of each user. (각 사용자의 확인률을 찾아라)
Return the result table in any order. */


SELECT s.user_id, round(avg(if(c.action="confirmed",1,0)),2) as confirmation_rate
-- 'confirmed'면 1, 아니면 0으로 바꿔 평균을 구한 뒤, 소수점 둘째 자리까지 반올림 → 확인율 계산
FROM Signups  s 
LEFT JOIN Confirmations  c   -- 가입한 사용자 중 일부만 확인했을 수 있으므로 LEFT JOIN
ON s.user_id= c.user_id 
GROUP BY user_id;  -- 사용자별로 그룹화하여 평균(확인율)을 계산



--3220. Odd and Even Transactions
/*문제: Write a solution to find the sum of amounts for odd and even transactions for each day. If there are no odd or even transactions for a specific date, display as 0. (홀수 및 짝수 거래의 금액 합계를 구해라. 특정날짜에 홀수 또는 짝수 거래가 없는 경우 0을 반환)
Return the result table ordered by transaction_date in ascending order. (transaction_date오름차순으로 정렬)*/


SELECT transaction_date, 
SUM(CASE WHEN amount%2 != 0 THEN amount ELSE 0 END) AS odd_sum, --홀수 합계
SUM(CASE WHEN amount%2 = 0 THEN amount ELSE 0 END) AS even_sum --짝수 합계
FROM transactions
GROUP BY transaction_date
ORDER BY transaction_date;




