-- 1164. Product Price at a Given Date
/*문제: Initially, all products have price 10. (처음가격은 10)
Write a solution to find the prices of all products on the date 2019-08-16. (19/08/16에 가격 구하기)
Return the result table in any order. */

SELECT product_id, 10 AS price
FROM Products
GROUP BY product_id
HAVING MIN(change_date) > '2019-08-16'
UNION ALL
SELECT product_id, new_price AS price
FROM Products
WHERE (product_id, change_date) IN (
  SELECT product_id, MAX(change_date)
  FROM Products
  WHERE change_date <= '2019-08-16'
  GROUP BY product_id)


-- 1174. Immediate Food Delivery II
/*문제: If the customer's preferred delivery date is the same as the order date, then the order is called immediate; otherwise, it is called scheduled. (고객이 선호하는 날짜=배송날짜 -> 주문을 즉시 처리, !=>예약처리)
The first order of a customer is the order with the earliest order date that the customer made. It is guaranteed that a customer has precisely one first order. (고객의 첫번째 주문은 고객이 가장 먼저 주문한 날짜가 있는 주문임. 고객은 정확히 한번의 첫번째 주문을 가지고 있음)
Write a solution to find the percentage of immediate orders in the first orders of all customers, rounded to 2 decimal places. (모든 고객의 첫번째 주문에서 즉시 주문의 비율을 구해라. 소수점 2자리까지 반올림)*/

/*1. 고객별 첫번째 주문 날짜 필터링하기
2. 첫번째 날짜=선호날짜 같은 수 찾기
3. 모든 고객의 첫번째 주문에서 즉시 주문의 비율 비율찾기 */

--sol1 비율을 avg()사용해서 구함
--order_date = customer_pref_delivery_date-> TRUE=1, FALSE=0임
--avg()사용하여 전체첫번째주문중 선호날짜랑 일치한 비율을 계산함

SELECT round(avg(order_date = customer_pref_delivery_date)*100, 2) AS immediate_percentage 
FROM Delivery
WHERE (customer_id, order_date) in (
  SELECT customer_id, min(order_date) 
  FROM Delivery
  FROM BY customer_id
);


--sol2 비율을 case문 사용해서 구함

SELECT round(100.0 * SUM(CASE WHEN order_date = customer_pref_delivery_date THEN 1 ELSE 0 END) / COUNT(*), 2  AS immediate_percentage
FROM Delivery
WHERE (customer_id, order_date) IN (
    SELECT customer_id, MIN(order_date)
    FROM Delivery
    FROM BY customer_id
);

-- 1193. Monthly Transactions I
/*문제:Write an SQL query to find for each month and country, the number of transactions and their total amount, the number of approved transactions and their total amount. (월별 및 국가별로 거래 수와 총액, 승인된 거래 수와 총액을 찾아라
Return the result table in any order. */

/*1. 월별, 국가별로 그룹바이하기
2. select절에서 집계하기 */ 
 -- trans_date 문자열의 앞 7글자('YYYY-MM')를 잘라 월 단위 키로 사용하고 별칭을 month로 지정

SELECT 
    LEFT(trans_date, 7) AS month,
    country, 
    COUNT(id) AS trans_count,
    SUM(state = 'approved') AS approved_count, --TRUE=1, FALSE=0 이라 합계가 곧 건수가 됨
    SUM(amount) AS trans_total_amount,
    SUM((state = 'approved') * amount) AS approved_total_amount
FROM 
    Transactions
GROUP BY 
    month, country;

/* COUNT(*): 해당 그룹의 모든 행을 센다(열 값이 NULL이어도 카운트) / COUNT(id): id가 NULL이 아닌 행만 센다.
id가 기본키(=NULL 불가) 라면 두 값은 동일하다. 이 경우 가독성/관례상 COUNT(*)를 많이 씀. */ 

-- 1204. Last Person to Fit in the Bus
-- 문제: There is a queue of people waiting to board a bus. However, the bus has a weight limit of 1000 kilograms, so there may be some people who cannot board. (버스의 무게 제한이 1000kg임. 탑승 못하는 사람 생김)
Write a solution to find the person_name of the last person that can fit on the bus without exceeding the weight limit. The test cases are generated such that the first person does not exceed the weight limit. (무게 제한을 초과하지 않고 버스에 탈 수 있는 마지막 사람의 person_name을 찾아라. 
Note that only one person can board the bus at any given turn. (버스는 한 번의 턴에 한 사람만 탑승할 수 있음)

/*1. turn을 오름차순으로 정렬 후 무게의 누적합을 구함
2. 누적합 무게가 1000을 초과하지 않는 마지막 사람을 필터링해서 이름 출력*/

--sol1 셀프조인을 통해 누적합 구하기

SELECT 
    q1.person_name
FROM Queue q1 
JOIN Queue q2               -- 같은 테이블을 q2로 조인
ON q1.turn >= q2.turn      -- q1의 turn보다 작거나 같은 모든 q2를 매칭 (->q1까지의 누적 범위)
GROUP BY q1.turn           -- q1.turn 별로 그룹화 → q1의 각 사람까지의 누적 그룹
HAVING SUM(q2.weight) <= 1000
ORDER BY SUM(q2.weight) DESC
LIMIT 1                          --맨 위 1명만 선택

--sol2 CTE , 윈도우함수로 누적합 임시 테이블 만들기
-- SUM() OVER(ORDER BY turn) 으로 바로 누적합을 계산

WITH new_table AS (
    SELECT 
        person_name,
        SUM(weight) OVER (ORDER BY turn) AS cumulative_weight   -- turn 순서대로 weight 누적합을 계산
    FROM Queue
)
SELECT 
    person_name
FROM new_table
WHERE cumulative_weight <= 1000
ORDER BY cumulative_weight DESC
LIMIT 1;

-- 1321. Restaurant Growth
/*문제: You are the restaurant owner and you want to analyze a possible expansion (there will be at least one customer every day). (레스토랑 가능한 확장 분석할거임. 매일 최소 한 명의 고객있음)
Compute the moving average of how much the customer paid in a seven days window (i.e., current day + 6 days before). average_amount should be rounded to two decimal places. (고객이 7일동안 지불한 금액의 이동평균을 계산해라. average_amount 소수점 두 자리로 반올림) 
Return the result table ordered by visited_on in ascending order. (visited_on 오름차순 정렬)*/

--sol1 윈도우함수
-- 현재 날짜 기준으로 이전 6일~현재까지의 기간(=7일간) 누적합 계산
-- 첫 방문일과의 날짜 차이를 계산(처음 6일은 제외)

SELECT visited_on, amount, average_amount 
FROM 
(SELECT DISTINCT visited_on, -- 중복된 날짜 제거 (없어도됨)
SUM(amount) OVER 
 (ORDER BY visited_on RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW) AS amount,
  ROUND(SUM(amount) OVER (ORDER BY visited_on RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW)/7,2)
   AS average_amount
FROM Customer) as whole_totals
WHERE DATEDIFF(visited_on, (SELECT MIN(visited_on) FROM Customer)) >= 6 

--sol2 공통 윈도우 정의를 미리 작성해두고 OVER w로 재사용
select distinct visited_on,
        sum(amount) over w as amount,
        round((sum(amount) over w)/7, 2) as average_amount
from customer
WINDOW w AS ( 
            order by visited_on
            range between interval 6 day PRECEDING and current row )
Limit 6, 999  -- 결과 출력 시 처음 6행은 건너뛰고 7번째 행부터 출력

--LIMIT[offset], [row_count] : [건너뛸 행의 개수], [그 다음부터 보여줄 최개 행의 개수]
/*윈도우함수: 
윈도우함수(인수) OVER (
    [ PARTITION BY partition_cols ]
    [ ORDER BY order_cols ]
    [ WINDOWING 절: ROWS / RANGE … ]
)
PARTITION BY : 데이터를 그룹 단위로 나눔 (그룹별 따로 계산)
ORDER BY : 파티션 내에서 정렬 기준
WINDOWING (또는 범위 지정 절) : 현재 행 기준 어느 범위를 볼지 지정 (예: 앞의 5행까지 등)
WINDOWING 절 종류
ROWS BETWEEN … : 물리적 행 수 기준
RANGE BETWEEN … : 값의 범위 기준 */ 





