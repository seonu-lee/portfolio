-- 608. Tree Node
-- 문제: Each node in the tree can be one of three types:
"Leaf": if the node is a leaf node.
"Root": if the node is the root of the tree.
"Inner": If the node is neither a leaf node nor a root node.
Write a solution to report the type of each node in the tree. (각각의 노드의 유형을 찾아라)
Return the result table in any order.

/* 1. 먼저 p_id IS NULL을 검사해서 Root로 분류함 (CASE 문은 위에서부터 순서대로 평가됨)
2. p_id에 존재하는 id는 inner로 분류함
3. 나머지는 leaf로 분류함) */

SELECT id,
    CASE 
        WHEN p_id IS NULL THEN 'Root' 
        WHEN id IN (SELECT p_id FROM Tree)THEN 'Inner' 
        ELSE 'Leaf'
        END AS type
 FROM Tree
	

-- 626. Exchange Seats
-- 문제: Write a solution to swap the seat id of every two consecutive students. If the number of students is odd, the id of the last student is not swapped. (연속되는 두 학생의 좌석id를 바꿔라, 학생 수가 홀수인 경우 마지막 학생의 id는 변경되지 않음)
Return the result table ordered by id in ascending order.(id의 오름차순 정렬)

--sol1: CASE문 사용, 홀수와 짝수의 id를 바꿔주고, id로 오름차순 정렬하기

SELECT 
    CASE 
        WHEN id % 2 = 0 THEN id - 1
	WHEN id % 2 = 1 AND id + 1 <= (SELECT MAX(id) FROM Seat) THEN id + 1
        ELSE id
    END AS id,
    student
FROM Seat
ORDER BY id;

--sol2: 윈도우함수사용해서 홀수와 짝수 id의 좌석 바꿔주기, 정렬 필요없음
/*  LAG(student) OVER(ORDER BY id): 현재 행을 기준으로 이전 행의 값을 가져옴
LEAD(student) OVER(ORDER BY id): 현재 행을 기준으로 다음 행의 값을 가져옴
COALESCE(a, b): null이 아닌 최소의 표현식 반환 (마지막 학생인 경우 null값 반환을 막기 위해 사용함)*/ 

SELECT 
    id,
    CASE
        WHEN id % 2 = 0 THEN LAG(student) OVER(ORDER BY id) 
        ELSE COALESCE(LEAD(student) OVER(ORDER BY id), student) 
    END AS student
FROM Seat


-- 1045. Customers Who Bought All Products
-- 문제: Write a solution to report the customer ids from the Customer table that bought all the products in the Product table. (고객테이블에서 제품테이블의 모든 제품을 구매한 고객 id를 구해라)
Return the result table in any order.

/* 고객id별 고유한 제품 수= 제품테이블의 제품 수인 id 찾기*/

SELECT  customer_id 
FROM Customer 
GROUP BY customer_id
HAVING COUNT(distinct product_key) = (SELECT COUNT(*) FROM Product)


-- 1070. Product Sales Analysis III
-- 문제: Write a solution to find all sales that occurred in the first year each product was sold.
(각 제품이 판매된 첫 해에 발생한 모든 판매를 찾아라)
For each product_id, identify the earliest year it appears in the Sales table.
Return all sales entries for that product in that year. (해당 연도에 해당 제품에 대한 모든 판매)
Return a table with the following columns: product_id, first_year, quantity, and price.
Return the result in any order. 

--sol1: where절에 필터링
select product_id, year as first_year, quantity, price 
from sales 
where (product_id, year) in (
    select product_id, min(year)
    from sales
    group by product_id)

--sol2 
/*1. 제품별 판매된 첫해 테이블 만들기
2. 조인 조건에 제품아이디, 연도 일치 조건을 넣어서 조인 첫해 테이블과 판매테이블 조인 */
--CTE사용
WITH firstyear AS (
  SELECT product_id, MIN(year) AS year
  FROM Sales
  GROUP BY product_id
)
SELECT s.product_id,
       s.year AS first_year,
       s.quantity,
       s.price
FROM firstyear f
JOIN Sales s
  ON f.product_id = s.product_id  
 AND f.year = s.year;    
--서브쿼리사용
SELECT s.product_id,
       s.year AS first_year,
       s.quantity,
       s.price
FROM (SELECT product_id, MIN(year) AS year
  FROM Sales
  GROUP BY product_id) AS f
JOIN Sales s
  ON f.product_id = s.product_id   
 AND f.year = s.year; 


-- 1158. Market Analysis I
-- 문제: Write a solution to find for each user, the join date and the number of orders they made as a buyer in 2019.
(2019 구매자로서 각 user, join date, 주문 수를 찾아라)
Return the result table in any order.

--sol1
/* 1. Users 테이블과 Orders 테이블을 user_id와 buyer_id로 조인함
2. id별로 그룹핑하고 case문을 활용해서 2019년도가 아니면 null값을 넣어서 count를 계산하게 함*/

select U.user_id AS buyer_id, 
	U.join_date AS join_date, 
	count(case when O.order_date between '2019-01-01' and '2019-12-31' then O.order_date
	else NULL END ) AS orders_in_2019
from Users U
join Orders O
on U.user_id = O.buyer_id 
group by O.buyer_id 

--sol2: on절에서 buyer_id랑 2019년만 조인하기
SELECT u.user_id as buyer_id, u.join_date, count(o.order_id) as 'orders_in_2019'
FROM Users u
LEFT JOIN Orders o
ON o.buyer_id=u.user_id AND YEAR(order_date)='2019'
GROUP BY u.user_id -- my sql만 가능함, sql표준모드에서는 비집계칼럼이 그룹바이에 다 들어가 있어야 함





