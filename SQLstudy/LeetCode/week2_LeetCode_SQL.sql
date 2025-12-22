--LeetCode176. Second Highest Salary

--Write a solution to find the second highest distinct salary from the Employee table. If there is no second highest salary, return null (return None in Pandas).

--sol1 (OFFSET 사용)

select(select distinct Salary 
from Employee order by salary desc 
limit 1 offset 1)
AS SecondHighestSalary;

--sol2

SELECT MAX(salary) AS SecondHighestSalary FROM Employee
WHERE salary < (SELECT MAX(salary) FROM Employee);

--sol3(OFFSET 사용X)

select IFNULL ( (
select salary from ( 
select ROW_NUMBER() OVER (ORDER BY salary desc) as row_num, salary 
from( 
    select   DISTINCT  salary from employee order by salary desc
    ) x
) y
where row\_num = 2 ), NULL ) AS SecondHighestSalary



--LeetCode177. Nth Highest Salary

--Write a solution to find the nth highest distinct salary from the Employee table. If there are less than n distinct salaries, return null.

--sol1

CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
SET N = N - 1;
RETURN (
SELECT DISTINCT salary
FROM Employee
ORDER BY salary DESC
LIMIT 1 OFFSET N
);

END



--LeetCode178. Rank Scores

--Write a solution to find the rank of the scores. The ranking should be calculated according to the following rules: The scores should be ranked from the highest to the lowest.
If there is a tie between two scores, both should have the same ranking.
After a tie, the next ranking number should be the next consecutive integer value. In other words, there should be no holes between ranks.
Return the result table ordered by score in descending order.

--sol1 윈도우함수사용

/*ranking 윈도우함수: 
RANK()-동일한 순위 부여(1, 1, 3, 4)
DENSE_RANK()-동일한 순위 부여,구멍없음(1, 1, 2, 3)
ROW_NUMBER()-고유한 순위부여(1,2,3,4) */

SELECT score,
DENSE_RANK() OVER (ORDER BY score DESC) AS `rank` 
--rank는 예약어이기 때문에 ``(백틱), ""(별칭에만 큰따옴표),\[](대괄호) 사용해야함
FROM Scores;

--sol2 윈도우함수X , 상호연관된 서브쿼리사용

SELECT s1.score, 
(SELECT COUNT(DISTINCT s2.score) 
FROM Scores s2 
WHERE s2.score >= s1.score) AS `rank`
--3.85점에 대해 이 서브쿼리를 실행하면 3.85점과 4점이 선택되어 2라는 순위가 계산됨
FROM Scores s1
ORDER BY s1.score DESC;

--sol3 self-join사용

SELECT s1.score, COUNT(DISTINCT s2.score) AS rank
--각 그룹 내에서 조인된 s2의 고유한(DISTINCT) 점수 개수가 순위가 됨
FROM Scores s1
LEFT JOIN Scores s2 ON s1.score <= s2.score
--s1의 점수가 3.85점인 행은 s2 테이블의 4점과 3.85점 행과 조인됨
GROUP BY s1.score
--s1.score를 기준으로 그룹화
ORDER BY s1.score DESC;



--LeetCode180. Consecutive Numbers

--Find all numbers that appear at least three times consecutively.
Return the result table in any order.

--sol1 self-join사용, 3번 셀프조인함

SELECT DISTINCT l1.num AS ConsecutiveNums
FROM Logs l1, Logs l2, Logs l3
WHERE l1.num = l2.num
  AND l2.num = l3.num
  AND l1.id = l2.id - 1
  AND l2.id = l3.id - 1;

--sol2 lead()윈도우함수이용
--with cte as (...) 구문: cte라는임시테이블 만듦, 메인쿼리에서 테이블처럼 사용할 수 있음
with cte as (
    select num,
    lead(num,1) over() num1,
    lead(num,2) over() num2
    from logs)
select distinct num ConsecutiveNums 
from cte where (num=num1) and (num=num2)

/*LEAD(column_name, offset, default_value) OVER (PARTITION BY ... ORDER BY ...)
column_name: 값을 가져올 열의 이름
offset: 몇 번째 뒤의 행을 가져올지 지정하는 숫. 기본값은 1.
default_value: 지정된 오프셋에 해당하는 행이 없을 때 반환할 값. 기본값은 NULL.
OVER (PARTITION BY ... ORDER BY ...): 윈도우(Window) 또는 파티션을 정의.
ORDER BY: 행의 순서를 결정합니다. 
PARTITION BY: 각 파티션 내에서 LEAD() 함수가 독립적으로 적용.
LEAD()와 반대되는 개념으로는 LAG() 함수*/



--LeetCode184. Department Highest Salary

--Write a solution to find employees who have the highest salary in each of the departments.
Return the result table in any order.

--sol1 서브쿼리
--1. 각 부서의 최고 연봉 구하기
--2. 최고연봉을 받는 직원정보 가져오기
SELECT
    D.name AS Department,
    E.name AS Employee,
    E.salary AS Salary
FROM
    Employee E
JOIN
    (SELECT departmentId, MAX(salary) AS max_salary
     FROM Employee
     GROUP BY departmentId) AS DepartmentMaxSalaries
ON
    E.departmentId = DepartmentMaxSalaries.departmentId AND E.salary = DepartmentMaxSalaries.max_salary
JOIN
    Department D
ON
    E.departmentId = D.id;

--sol2 CTE&윈도우함수
WITH RankedEmployees AS (
    SELECT *,
        DENSE_RANK() OVER(PARTITION BY departmentId ORDER BY salary DESC) as rank_num
    FROM Employee)

SELECT
    D.name AS Department,
    RE.name AS Employee,
    RE.salary AS Salary
FROM
    RankedEmployees RE
JOIN
    Department D ON RE.departmentId = D.id
WHERE
    RE.rank_num = 1;



--LeetCode185. Department Top Three Salaries (Hard)

--A company's executives are interested in seeing who earns the most money in each of the company's departments. A high earner in a department is an employee who has a salary in the top three unique salaries for that department.
Write a solution to find the employees who are high earners in each of the departments.
Return the result table in any order.

--sol1 서브쿼리
SELECT
    D.name AS Department,
    RE.name AS Employee,
    RE.salary AS Salary
FROM
    (SELECT *, DENSE_RANK() OVER(PARTITION BY departmentId ORDER BY salary DESC) as rank_num
     FROM Employee) AS RE
JOIN
    Department D
ON
    RE.departmentId = D.id
WHERE
    RE.rank_num <= 3;


--sol2 CTE&윈도우함수
WITH RankedEmployees AS (
    SELECT *,
        DENSE_RANK() OVER(PARTITION BY departmentId ORDER BY salary DESC) as rank_num
    FROM Employee)

SELECT
    D.name AS Department,
    RE.name AS Employee,
    RE.salary AS Salary
FROM
    RankedEmployees RE
JOIN
    Department D ON RE.departmentId = D.id
WHERE
    RE.rank_num <= 3;






