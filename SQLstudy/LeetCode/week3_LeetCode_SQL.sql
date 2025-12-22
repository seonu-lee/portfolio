-- 262. Trips and Users(hard)
-- 문제: Write a solution to find the cancellation rate of requests with unbanned users (both client and driver must not be banned) each day between "2013-10-01" and "2013-10-03" with at least one trip. Round Cancellation Rate to two decimal points.(2013-10-01부터 2013-10-03까지 각각 최소 한 번 이상 여행을 한 금지되지 않은 사용자의 요청 취소율을 찾아라, 소수점 두 자리까지)
--The cancellation rate is computed by dividing the number of canceled (by client or driver) requests with unbanned users by the total number of requests with unbanned users on that day.(금지 되지않은 사용자가 취소한 요청 수 /당일 금지되지 않은 사용자의 총 요청 수)

-- sol1
/*1. CTE를 사용하여 User테이블에서 금지된 모든 user ID를 가져옴
2. client_id 또는 driver_id가 금지된 모든 여행을 필터링함
3. 데이터를 날짜범위로 제한함
4. 데이터를 각 날짜별로 그룹화함
5. 각 그룹에 대해 status가 completed가 아닌 여행 수를 1로만들어서 다 더함(수를 셈), 그것을 그 날의 총 여행 수로 나눔
6. 최종 취소율을 소수점 두 자리까지로 반올림함*/

WITH banned_users AS (
    SELECT users_id
    FROM Users
    WHERE banned = "Yes"
)

SELECT 
    request_at AS Day, 
    ROUND(SUM(IF(status != "completed", 1, 0)) / COUNT(*), 2) AS "Cancellation Rate"
FROM Trips
WHERE 
    request_at BETWEEN "2013-10-01" AND "2013-10-03"
    AND client_id NOT IN (SELECT * FROM banned_users)
    AND driver_id NOT IN (SELECT * FROM banned_users)
GROUP BY request_at;

--sol2
/*1. Trip 테이블과 Users 테이블을 banned가 NO인 id로 조인함.
2. 데이터를 날짜범위로 제한함
3. 데이터를 각 날짜별로 그룹화함
4.  각 그룹에 대해 status가 completed가 아닌 여행 수를 더함, 그것을 그 날의 총 여행 수로 나눔
5.  최종 취소율을 소수점 두 자리까지로 반올림함*/

SELECT 
    request_at AS Day,             
    ROUND(
        SUM(status != 'completed') / COUNT(*), 2
    ) AS "Cancellation Rate"      
FROM Trips t
JOIN Users c 
    ON t.client_id = c.users_id AND c.banned = 'No'   
JOIN Users d 
    ON t.driver_id = d.users_id AND d.banned = 'No'   
WHERE request_at BETWEEN '2013-10-01' AND '2013-10-03' GROUP BY request_at;            


-- 550. Game Play Analysis IV
-- 문제: Write a solution to report the fraction of players that logged in again on the day after the day they first logged in, rounded to 2 decimal places. In other words, you need to determine the number of players who logged in on the day immediately following their initial login, and divide it by the number of total players. (처음 로그인한 다음 날에 다시 로그인한 플레이어의 비율을 구해라. 소수점 두 번째자리까지 반올림 -> 최초 로그인 직후에 다시 로그인한 플레이어수 / 총 플레이어 수)

--sol1 
/*1. CTE를 사용하여 첫번째 로그인한 플레이어 찾기
2. CTE를 사용하여 다음 날 다시 로그인한 플레이어 찾기, DATE_ADD: 날짜 더하기 함수
3. 다음 날 로그인한 플레이어 수에서 총 플레이어 수 나누고 소수점 두 번째 자리까지 반올림하기*/

WITH FirstLogin AS (
    SELECT
        player_id,
        MIN(event_date) AS first_login
    FROM Activity
    GROUP BY player_id
),
ConsecutiveLogin AS (
    SELECT
        a.player_id
    FROM Activity a
    JOIN FirstLogin fl
    ON a.player_id = fl.player_id
    AND a.event_date = DATE_ADD(fl.first_login, INTERVAL 1 DAY)
)
SELECT
    ROUND(
        COUNT(DISTINCT cl.player_id)  / COUNT(DISTINCT fl.player_id),
        2
    ) AS fraction
FROM FirstLogin fl
LEFT JOIN ConsecutiveLogin cl
ON fl.player_id = cl.player_id;

--sol2
/* 1. 전날날짜랑 처음 로그인한 날짜랑 같은 플레이어 필터링하기, DATE_SUB: 날짜 빼기 함수
2. 필터링한 플레이어 수에서 총 플레이어 수를 나누고, 소수점 두 번째 자리까지 반올림하기*/

SELECT
  ROUND(COUNT(DISTINCT player_id) / (SELECT COUNT(DISTINCT player_id) FROM Activity), 2) AS fraction
FROM
  Activity
WHERE
  (player_id, DATE_SUB(event_date, INTERVAL 1 DAY))
  IN (
    SELECT player_id, MIN(event_date) AS first_login FROM Activity GROUP BY player_id
  )


--570. Managers with at Least 5 Direct Reports
-- 문제: Write a solution to find managers with at least five direct reports.

--sol1 INNER JOIN
SELECT M.name
FROM Employee E
INNER JOIN Employee M
ON E.managerId = M.id
GROUP BY E.managerId
HAVING count(E.managerId)>=5

--sol2 LEFT JOIN
SELECT E.name
FROM employee E
LEFT JOIN employee M 
ON E.id=M.managerId
GROUP BY E.id
HAVING COUNT(M.name) >= 5;

--sol3 JOIN
SELECT a.name 
FROM Employee a 
JOIN Employee b ON a.id = b.managerId 
GROUP BY b.managerId 
HAVING COUNT(*) >= 5

--sol4 서브쿼리
SELECT E1.name
FROM Employee E1
JOIN (
    SELECT managerId, COUNT(*) AS directReports
    FROM Employee
    GROUP BY managerId
    HAVING COUNT(*) >= 5
) E2 
ON E1.id = E2.managerId;


--585. Investments in 2016
-- 문제: Write a solution to report the sum of all total investment values in 2016 tiv_2016, for all policyholders who:
have the same tiv_2015 value as one or more other policyholders, and
are not located in the same city as any other policyholder (i.e., the (lat, lon) attribute pairs must be unique).
Round tiv_2016 to two decimal places. (2016년에 모든 총 투자 가치의 합계를 구하라, 하나 이상의 다른 보험 가입자와 동일한 tiv_2015값을 가짐, 다른 보험 가입자와 같은 도시에 위치하지 않음 (lat, lon)쌍이 고유해야함)

--sol1
/*1. 2015년의 보험가치가 두 번 이상인 행 찾음
2. 위도와 경도 쌍이 하나만 존재하는 행을 찾음
3. 위 두 가지 조건을 필터링 함
4. 해당하는 행의 2016년의 보험가치의 합을 구하고, 소수점 두 번째 자리까지 반올림함*/

SELECT ROUND(SUM(tiv_2016), 2) AS tiv_2016
FROM Insurance
WHERE tiv_2015 IN (
    SELECT tiv_2015
    FROM Insurance
    GROUP BY tiv_2015
    HAVING COUNT(*) > 1
)
AND (lat, lon) IN (
    SELECT lat, lon
    FROM Insurance
    GROUP BY lat, lon
    HAVING COUNT(*) = 1
)

--sol2
/* 1. CTE사용해서 조건에 필요한 칼럼을 넣은 임의 테이블 만들기
2. 조건을 필터링 함
3. 2016년의 보험가치의 합을 구하고, 소수점 두 번째 자리까지 반올림함*/

WITH uniq_coords AS (
  SELECT *, 
    COUNT(*) OVER (PARTITION BY lat, lon) AS attempts,
    COUNT(*) OVER (PARTITION BY tiv_2015) AS tivs
  FROM Insurance
)

SELECT ROUND(SUM(tiv_2016)::numeric, 2) AS tiv_2016
FROM uniq_coords
WHERE attempts = 1 AND tivs > 1;


--601. Human Traffic of Stadium(hard)
-- 문제: Write a solution to display the records with three or more rows with consecutive id's, and the number of people is greater than or equal to 100 for each. Return the result table ordered by visit_date in ascending order. (연속된 ID를 가진 세 개 이상의 행으로 기록을 표시해라. 각 행에 대해 100명 이상의 인원을 배치해라. visit_date로 정렬, 오름차순)

/* 1. 관객 수가 100명 이상인 행만 필터링 함
2. 윈도우함수를 사용해 현재 행을 기준으로 연속된 행의 개수를 계산함
3. 위의 행의 개수 중에 3인 행을 필터링 
4. 날짜로 정렬, 기본값이 ASC*/

--sol1 
WITH q1 AS (
SELECT *, 
     count(*) over( order by id range between current row and 2 following ) following_cnt, -- id의 현재 행부터 바로 뒤의 2개 행까지 포함한 행의 개수
     count(*) over( order by id range between 2 preceding and current row ) preceding_cnt, --id의 앞의 2개 행부터 현재 행까지 포함한 행의 개수
     count(*) over( order by id range between 1 preceding and 1 following ) current_cnt --id의 현재 행을 중심으로 앞뒬ㅗ 1개씩 포함한 행의 개수
FROM stadium
WHERE people > 99
)
SELECT id, visit_date, people
FROM q1
WHERE following_cnt = 3 or preceding_cnt = 3 or current_cnt = 3
ORDER BY visit_date

/* RANGE / ROWS BETWEEN: ROWS는 물리적인 행 기준,  RANGE는 정렬된 컬럼 값 기준(값이 같으면 묶여서 다 포함됨.)*/

--sol2
/*1. 관객 수가 100명 이상인 행만 필터링
2. row_number() over(order by id)로 쿼리결과에서 각 행에 순번을 매김
3. 연속된 값들은 id - row_number 값이 동일한 값을 가짐
4. 내부 서브쿼리를 이용해  id_diff별로 몇 개의 행이 있는 지 세고, 3개 이상인 그룹만 필터링
5. 날짜로 정렬, 기본값이 ASC*/

WITH  q1 AS (
SELECT *, id - row_number() over(order by id) as id_diff
FROM stadium
WHERE people > 99
)
SELECT id, visit_date, people
FROM q1
WHERE id_diff 
IN (SELECT id_diff FROM q1 GROUP BY id_diff HAVING count(*) > 2)
ORDER BY visit_date

--602. Friend Requests II: Who Has the Most Friends
-- 문제: Write a solution to find the people who have the most friends and the most friends number.
The test cases are generated so that only one person has the most friends. (가장 많은 친구 수를 가진 사람들 찾기, test cases는 한 사람만이 가장 많은 친구를 가질 수 있도록 생성됨)

/*1. UNION ALL을 이용하여 모든 사람 id가 한 열에 모여 있는 테이블  all_ids을 만듦
2. id별로 그룹화해서 행의 수(등장 수)를 셈
3. 등장 수= 친구 수를 기준으로 내림차순, limit 1을 통해 가장 많은 친구를 가진 1명만 출력*/

SELECT id, COUNT(*) AS num
FROM (
    SELECT requester_id AS id FROM RequestAccepted
    UNION ALL  --중복 제거를 하지 않고 그대로 합쳐야 각 사람의 등장 횟수를 정확히 셀 수 있음
    SELECT accepter_id AS id FROM RequestAccepted
) AS all_ids
GROUP BY id
ORDER BY num DESC
LIMIT 1;

/* UNION은 중복된 행을 제거하고 유일한 행만 보여주는 반면, UNION ALL은 중복을 포함하여 모든 행을 그대로 보여줌*/



