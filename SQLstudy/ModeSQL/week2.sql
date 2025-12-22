--1. SQL Aggregate Functions (집계함수)
/*COUNT: 특정 열에 몇 개의 행이 있는지 계산합니다.
SUM:특정 열의 모든 값을 더합니다.
MIN, MAX: 각각 특정 열의 가장 낮은 값과 가장 높은 값을 반환합니다.
AVG: 선택된 값 그룹의 평균을 계산합니다. 
산술 연산자는 행에 대해서만 연산을 수행함->집계 함수는 전체 열에 대해 연산을 수행함 */
--SELECT * FROM tutorial.aapl_historical_stock_price 사용

--2. SQL COUNT
--모든 행 계산
SELECT COUNT(*)
  FROM tutorial.aapl_historical_stock_price --COUNT(1)과 같은 결과를 냄, null이 포함
--개별 열 계산
SELECT COUNT(high)
  FROM tutorial.aapl_historical_stock_price --high열이 null이 아닌 모든 행의 개수를 계산
--연습문제:Write a query to count the number of non-null rows in the low column.
SELECT COUNT(low) 
  FROM tutorial.aapl_historical_stock_price
--숫자가 아닌 열 계산
SELECT COUNT(date) AS count_of_date --보기편하게 열 이름을 지정하는 것이 좋음
  FROM tutorial.aapl_historical_stock_price
--공백을 사용해야 하는 경우에는 유일하게 큰따옴표를 사용, 그 외의 경우에는 작은따옴표를 사용
SELECT COUNT(year) AS year,
       COUNT(month) AS month,
       COUNT(open) AS open,
       COUNT(high) AS high,
       COUNT(low) AS low,
       COUNT(close) AS close,
       COUNT(volume) AS volume
  FROM tutorial.aapl_historical_stock_price

--3. SQL SUM
--SQL SUM 함수: 주어진 열의 값을 합산하는 SQL 집계 함수, count와 달리 숫자값이 포함된 열에만 사용가능
SELECT SUM(volume)
  FROM tutorial.aapl_historical_stock_price
--집계함수는 수직으로만 계산됨, 행에 걸쳐 계산하려면 간단한 산술연산을 사용해야함(high+low)
--SUM은 null 을 0으로 처리함
--연습문제:Write a query to calculate the average opening price (hint: you will need to use both COUNT and SUM, as well as some simple arithmetic.).
SELECT SUM(open)/count(open) AS avg_open
  FROM tutorial.aapl_historical_stock_price

--4. SQL MIN/MAX
--SQL MIN 및 MAX 함수:특정 열의 가장 낮은 값과 가장 높은 값을 반환하는 SQL 집계 함수
--숫자가 아닌 열에도 사용할 수 있다는 점에서 count와 유사
--MIN "A"에 알파벳순으로 가장 가까운 가장 낮은 숫자, 가장 빠른 날짜 또는 숫자가 아닌 값을 반환
SELECT MIN(volume) AS min_volume,
       MAX(volume) AS max_volume
  FROM tutorial.aapl_historical_stock_price
--연습문제:What was Apple's lowest stock price (at the time of this data collection)?, 그당시 최저가
SELECT MIN(low)
  FROM tutorial.aapl_historical_stock_price
--연습문제:What was the highest single-day increase in Apple's share value? 하루동안 가장 큰 상승
SELECT MAX(close - open)
  FROM tutorial.aapl_historical_stock_price
