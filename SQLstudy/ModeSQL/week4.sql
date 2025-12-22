-- 9. SQL DISTINCT
/* 1. SQL DISTINCT: 특정 열의 고유 값만 보고 싶을 때
2. 각 열의 고유 값을 살펴보면 데이터를 그룹화하거나 필터링하는 방법을 파악하는 데 도움됨
3.두 개 이상의 열을 포함하는 경우 해당 두 열의 모든 고유한 쌍이 포함됨
4. select절 시작 뿐만 아니라 집계함수 안에서도 사용됨*/

SELECT DISTINCT month
  FROM tutorial.aapl_historical_stock_price

--두 개 이상의 열을 포함하는 경우 해당 두 열의 모든 고유한 쌍이 포함됨
SELECT DISTINCT year, month
  FROM tutorial.aapl_historical_stock_price

--연습문제: Write a query that returns the unique values in the year column, in chronological order.
SELECT DISTINCT year
  FROM tutorial.aapl_historical_stock_price
 ORDER BY year

--집계에서 DISTINCT 사용: 함수와 함께 가장 많이 사용됨(count에서 많이 씀/ 하지만 sum,avg는 실용적이지 않음/ max,min은 있으나 없으나 똑같음)-> 쿼리속도가 느려짐 ㅜㅜ
SELECT COUNT(DISTINCT month) AS unique_months
  FROM tutorial.aapl_historical_stock_price

--연습문제: Write a query that counts the number of unique values in the month column for each year.
SELECT year,
       COUNT(DISTINCT month) AS months_count
  FROM tutorial.aapl_historical_stock_price
 GROUP BY year
 ORDER BY year

--연습문제: Write a query that separately counts the number of unique values in the month column and the number of unique values in the `year` column.
SELECT COUNT(DISTINCT year) AS years_count,
       COUNT(DISTINCT month) AS months_count
  FROM tutorial.aapl_historical_stock_price

--10. SQL Joins
/* 1. SQL 조인 소개: 관계형 개념:  "관계형 데이터베이스"라는 용어는 그 안의 테이블들이 서로 "관련"되어 있다는 사실을 의미->여러 테이블의 정보를 쉽게 결합할 수 있도록 하는 공통 식별자를 포함하고 있다는 것
2. SQL의 별칭: 조인을 수행할 때 테이블 이름에 별칭을 지정하는 것이 쉬움, 열 이름과 마찬가지로 모두 소문자, 공백 대신 언더바 사용하기
3. JOIN 및 ON: ON은 두 테이블이 서로 어떻게 관련되어있는지를 나타냄, 두 테이블을 연결하는 관계를 "매핑"이라고 함. 서로 매핑되는 두 열을 "외래키", "조인키" 라고 함. */ 

SELECT teams.conference AS conference,
       AVG(players.weight) AS average_weight
  FROM benn.college_football_players players
  JOIN benn.college_football_teams teams
    ON teams.school_name = players.school_name
 GROUP BY teams.conference
 ORDER BY AVG(players.weight) DESC

-연습문제: Write a query that selects the school name, player name, position, and weight for every player in Georgia, ordered by weight (heaviest to lightest). Be sure to make an alias for the table, and to reference all column names in relation to the alias.
SELECT players.school_name,
       players.player_name,
       players.position,
       players.weight
  FROM benn.college_football_players players
 WHERE players.state = 'GA'
 ORDER BY players.weight DESC

SELECT *
  FROM benn.college_football_players players
  JOIN benn.college_football_teams teams
    ON teams.school_name = players.school_name

--SELECT *는 뒤에 있는 테이블뿐만 아니라 두 테이블의 모든 열을 반환함 / 한 테이블의 열만 반환하려면 SELECT players.*

--11. SQL INNER JOIN
/* 1. INNER JOIN: 두 테이블에서 해당 명령문에 명시된 조인 조건을 충족하지 않는 행을 제거함, 수학적으로 내부 조인은 두 테이블의 교집합, join과 일치, 두 테이블 모두에서 일치하는 값만 보여줌
2. 동일한 열 이름을 가진 테이블 조인: 결과는 지정된 이름을 가진 하나의 열만 출력됨, 두 열 모두 출력도 가능하지만 동일한 데이터를 가짐*/

--연습문제: Write a query that displays player names, school names and conferences for schools in the "FBS (Division I-A Teams)" division.
SELECT players.player_name,
       players.school_name,
       teams.conference
  FROM benn.college_football_players players
  JOIN benn.college_football_teams teams
    ON teams.school_name = players.school_name
 WHERE teams.division = 'FBS (Division I-A Teams)'


--12. SQL OUTER JOIN
/* 1. LEFT JOIN(OUTER LEFT JOIN): 왼쪽 테이블에서 일치하지 않는 행만 반환하고 , 두 테이블에서 일치하는 행도 반환합니다.
RIGHT JOIN(OUTER RIGHT JOIN): 오른쪽 테이블에서 일치하지 않는 행만 반환하고 , 두 테이블에서 일치하는 행도 반환합니다.
FULL OUTER JOIN(OUTER JOIN): 두 테이블 모두에서 일치하지 않는 행 과 두 테이블 모두에서 일치하는 행을 반환합니다, 두 테이블의 합집합(union)*/

