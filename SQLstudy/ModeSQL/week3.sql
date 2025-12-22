--5. SQL AVG
--SQL AVG 함수: 선택한 값 그룹의 평균을 계산하는 SQL 집계 함수, 숫자 열에만 사용가능, Null 값을 완전히 무시
SELECT AVG(high)
  FROM tutorial.aapl_historical_stock_price
 WHERE high IS NOT NULL
SELECT AVG(high)
  FROM tutorial.aapl_historical_stock_price
-- 위에 두 쿼리의 결과가 같음(avg는 null을 무시하기 때문)
--연습문제:Write a query that calculates the average daily trade volume for Apple stock.
SELECT AVG(volume) AS avg_volume
  FROM tutorial.aapl_historical_stock_price

--6. SQL GROUP BY
--SQL GROUP BY 절: 테이블의 일부만 집계하고 싶을 때 사용
SELECT year,
       COUNT(*) AS count
  FROM tutorial.aapl_historical_stock_price
 GROUP BY year
여러 열로 그룹화할 수 있지만, ORDER BY와 마찬가지로 열 이름은 쉼표로 구분
--연습문제: Calculate the total number of shares traded each month. Order your results chronologically.
SELECT year,
       month,
       SUM(volume) AS volume_sum
  FROM tutorial.aapl_historical_stock_price
 GROUP BY year, month
 ORDER BY year, month
--열 번호로 그룹화
--ORDER BY와 마찬가지로 열 이름을 숫자로 대체 가능(텍스트가 지나치게 길어지는 경우에만 사용하기)
SELECT year,
       month,
       COUNT(*) AS count
  FROM tutorial.aapl_historical_stock_price
 GROUP BY 1, 2
--ORDER BY와 함께 GROUP BY 사용
--GROUP BY에서 열 이름의 순서는 중요하지 않음, 결과 동일함
--ORDER BY는 열 이름의 순서에 따라 정렬됨
SELECT year,
       month,
       COUNT(*) AS count
  FROM tutorial.aapl_historical_stock_price
 GROUP BY year, month
 ORDER BY month, year
--LIMIT과 함께 GROUP BY 사용
--SQL은 LIMIT절 전에 집계를 평가함 
--연습문제: Write a query to calculate the average daily price change in Apple stock, grouped by year.
SELECT year,
       AVG(close - open) AS avg_daily_change
  FROM tutorial.aapl_historical_stock_price
 GROUP BY 1
 ORDER BY 1
--연습문제: Write a query that calculates the lowest and highest prices that Apple stock achieved each month.
SELECT year,
       month,
       MIN(low) AS lowest_price,
       MAX(high) AS highest_price
  FROM tutorial.aapl_historical_stock_price
 GROUP BY 1, 2
 ORDER BY 1, 2

--7. SQL HAVING
--SQL HAVING 절: 집계 열을 필터링 함
SELECT year,
       month,
       MAX(high) AS month_high
  FROM tutorial.aapl_historical_stock_price
 GROUP BY year, month
HAVING MAX(high) > 400
 ORDER BY year, month
--쿼리 절 순서
/*SELECT
FROM
WHERE
GROUP BY
HAVING
ORDER BY
(LIMIT)*/

--8. SQL CASE
SELECT * FROM benn.college_football_players --사용
--SQL CASE 문: SQL에서 if/then 논리를 처리하는 방식
--CASE문장 뒤에는 적어도 한 쌍의 WHEN and THEN문이 옴
--모든 CASE명령문은 END명령문으로 끝나야함
SELECT player_name,
       year,
       CASE WHEN year = 'SR' THEN 'yes'
            ELSE NULL END AS is_a_senior
  FROM benn.college_football_players
--year가 SR이면 yes, 아니면 null
SELECT player_name,
       year,
       CASE WHEN year = 'SR' THEN 'yes'
            ELSE 'no' END AS is_a_senior
  FROM benn.college_football_players
--연습문제: Write a query that includes a column that is flagged "yes" when a player is from California, and sort the results with those players first. 
SELECT player_name,
       state,
       CASE WHEN state = 'CA' THEN 'yes'
            ELSE NULL END AS from_california
  FROM benn.college_football_players
 ORDER BY 3
--정렬할때 null값이 맨 마지막
--CASE 문에 여러 조건 추가
SELECT player_name,
       weight,
       CASE WHEN weight > 250 THEN 'over 250'
            WHEN weight > 200 THEN '201-250'
            WHEN weight > 175 THEN '176-200'
            ELSE '175 or under' END AS weight_group
  FROM benn.college_football_players
-- 효과적이긴 하지만 겹치지 않는 명령문을 만드는 것이 좋음
SELECT player_name,
       weight,
       CASE WHEN weight > 250 THEN 'over 250'
            WHEN weight > 200 AND weight <= 250 THEN '201-250'
            WHEN weight > 175 AND weight <= 200 THEN '176-200'
            ELSE '175 or under' END AS weight_group
  FROM benn.college_football_players
--연습문제: Write a query that includes players' names and a column that classifies them into four categories based on height. Keep in mind that the answer we provide is only one of many possible answers, since you could divide players' heights in many ways.
SELECT player_name,
       height,
       CASE WHEN height > 74 THEN 'over 74'
            WHEN height > 72 AND height <= 74 THEN '73-74'
            WHEN height > 70 AND height <= 72 THEN '71-72'
            ELSE 'under 70' END AS height_group
  FROM benn.college_football_players
--CASE 기본 사항에 대한 간략한 검토:
/* 1. 문장 CASE은 항상 SELECT절 에 들어갑니다.
2. CASE다음 구성 요소를 포함해야 합니다: WHEN, THEN, 및 END. ELSE는 선택적 구성 요소입니다.
3. WHEN와 THEN 사이에 있는 조건 연산자(예: WHERE) 를 사용하여 모든 조건문을 만들 수 있습니다. 여기에는 AND와OR를 사용하여 여러 조건문을 연결하는 것도 포함됩니다 .
4. 여러 개의 WHEN진술문을 포함할 수 있으며, 해결되지 않은 상황을 처리하기 위한 ELSE 진술문도 포함할 수 있습니다. */
--연습문제: Write a query that selects all columns from benn.college_football_players and adds an additional column that displays the player's name if that player is a junior or senior.
SELECT *,
       CASE WHEN year IN ('JR', 'SR') THEN player_name ELSE NULL END AS upperclass_player_name
  FROM benn.college_football_players
--집계 함수와 함께 CASE 사용
SELECT CASE WHEN year = 'FR' THEN 'FR'
            ELSE 'Not FR' END AS year_group,
            COUNT(1) AS count
  FROM benn.college_football_players
 GROUP BY CASE WHEN year = 'FR' THEN 'FR'
               ELSE 'Not FR' END
SELECT COUNT(1) AS fr_count
  FROM benn.college_football_players
 WHERE year = 'FR'
--WHERE절을 사용하면 조건을 하나만 계산할 수 있습니다. 다음은 하나의 쿼리에서 여러 조건을 계산하는 예입니다.
SELECT CASE WHEN year = 'FR' THEN 'FR'
            WHEN year = 'SO' THEN 'SO'
            WHEN year = 'JR' THEN 'JR'
            WHEN year = 'SR' THEN 'SR'
            ELSE 'No Year Data' END AS year_group,
            COUNT(1) AS count
  FROM benn.college_football_players
 GROUP BY 1
--숫자대신 열의 별칭도 사용가능
SELECT CASE WHEN year = 'FR' THEN 'FR'
            WHEN year = 'SO' THEN 'SO'
            WHEN year = 'JR' THEN 'JR'
            WHEN year = 'SR' THEN 'SR'
            ELSE 'No Year Data' END AS year_group,
            COUNT(1) AS count
  FROM benn.college_football_players
 GROUP BY year_group
--문장 전체를 반복하기로 선택한 경우 절 에 복사/붙여넣기할 때 열 이름을 제거
SELECT CASE WHEN year = 'FR' THEN 'FR'
            WHEN year = 'SO' THEN 'SO'
            WHEN year = 'JR' THEN 'JR'
            WHEN year = 'SR' THEN 'SR'
            ELSE 'No Year Data' END AS year_group,
            COUNT(1) AS count
  FROM benn.college_football_players
 GROUP BY CASE WHEN year = 'FR' THEN 'FR'
               WHEN year = 'SO' THEN 'SO'
               WHEN year = 'JR' THEN 'JR'
               WHEN year = 'SR' THEN 'SR'
               ELSE 'No Year Data' END
--연습문제: Write a query that counts the number of 300lb+ players for each of the following regions: West Coast (CA, OR, WA), Texas, and Other (everywhere else).
SELECT CASE WHEN state IN ('CA', 'OR', 'WA') THEN 'West Coast'
            WHEN state = 'TX' THEN 'Texas'
            ELSE 'Other' END AS arbitrary_regional_designation,
            COUNT(1) AS players
  FROM benn.college_football_players
 WHERE weight >= 300
 GROUP BY 1
--연습문제: Write a query that calculates the combined weight of all underclass players (FR/SO) in California as well as the combined weight of all upperclass players (JR/SR) in California.
SELECT CASE WHEN year IN ('FR', 'SO') THEN 'underclass'
            WHEN year IN ('JR', 'SR') THEN 'upperclass'
            ELSE NULL END AS class_group,
       SUM(weight) AS combined_player_weight
  FROM benn.college_football_players
 WHERE state = 'CA'
 GROUP BY 1
--집계 함수 내부에서 CASE 사용
--pivoting: 데이터를 가로로 표시하기 위해 세로->가로, 수평으로 바꿈
SELECT COUNT(CASE WHEN year = 'FR' THEN 1 ELSE NULL END) AS fr_count,
       COUNT(CASE WHEN year = 'SO' THEN 1 ELSE NULL END) AS so_count,
       COUNT(CASE WHEN year = 'JR' THEN 1 ELSE NULL END) AS jr_count,
       COUNT(CASE WHEN year = 'SR' THEN 1 ELSE NULL END) AS sr_count
  FROM benn.college_football_players
--연습문제: Write a query that displays the number of players in each state, with FR, SO, JR, and SR players in separate columns and another column for the total number of players. Order results such that states with the most players come first.
SELECT state,
       COUNT(CASE WHEN year = 'FR' THEN 1 ELSE NULL END) AS fr_count,
       COUNT(CASE WHEN year = 'SO' THEN 1 ELSE NULL END) AS so_count,
       COUNT(CASE WHEN year = 'JR' THEN 1 ELSE NULL END) AS jr_count,
       COUNT(CASE WHEN year = 'SR' THEN 1 ELSE NULL END) AS sr_count,
       COUNT(1) AS total_players
  FROM benn.college_football_players
 GROUP BY state
 ORDER BY total_players DESC
--연습문제:Write a query that shows the number of players at schools with names that start with A through M, and the number at schools with names starting with N - Z.
SELECT CASE WHEN school_name < 'n' THEN 'A-M'
            WHEN school_name >= 'n' THEN 'N-Z'
            ELSE NULL END AS school_name_group,
       COUNT(1) AS players
  FROM benn.college_football_players
 GROUP BY 1

