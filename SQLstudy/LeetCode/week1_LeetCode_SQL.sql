-- 1. Recyclable and Low Fat Products(easy)
-- 문제: Write a solution to find the ids of products that are both low fat and recyclable.
-- low_fat이 'Y'이고 recyclable이 'Y'인 제품의 product_id를 찾으세요.
-- 이 코드는 low_fat='Y'와 recyclable='Y'라는 두 조건을 모두 만족하는 제품의 ID를 반환합니다.

SELECT
    product_id
FROM
    Products
WHERE
    low_fat = 'Y' AND recyclable = 'Y';


-- 2. Customers Who Never Order(easy)
-- 방법 1: 서브쿼리 사용
-- 문제: Write a solution to find all customers who never order anything.
-- 주문 기록이 없는 고객의 이름을 찾으세요.
-- Orders 테이블에 ID가 없는 고객을 찾습니다.

SELECT
    name
FROM
    Customers
WHERE
    id NOT IN (SELECT customerId FROM Orders);

-- 방법 2: LEFT JOIN 사용
-- Customers 테이블에 Orders 테이블을 LEFT JOIN하고, 주문 기록이 없는 (customerId가 NULL인) 고객을 찾습니다.

SELECT
    c.name
FROM
    Customers c
LEFT JOIN
    Orders o ON c.id = o.customerId
WHERE
    o.customerId IS NULL;

-- 3. Find the Customer Referee(easy)
-- 문제: Find the names of the customer that are either: referred by any customer with id != 2. not referred by any customer.
-- referee_id가 2가 아닌 고객의 이름을 찾으세요. referee_id가 NULL인 경우도 포함해야 합니다.
-- 이 코드는 referee_id가 2가 아니거나 (OR) referee_id가 NULL인 고객을 찾습니다.
-- NULL 값은 = 또는 != 연산자로 비교할 수 없으므로 IS NULL을 사용해야 합니다.

SELECT
    name
FROM
    Customer

-- 4. Department Highest Salary(med)
-- 문제: Write a solution to find employees who have the highest salary in each of the departments.
-- 각 부서에서 가장 높은 급여를 받는 직원의 이름, 급여, 부서 이름을 찾으세요.
-- 이 코드는 각 부서의 최고 급여를 먼저 찾고, 해당 급여를 받는 직원을 JOIN하여 결과를 반환합니다.

SELECT
    d.name AS Department,
    e.name AS Employee,
    e.salary AS Salary
FROM
    Employee e
JOIN
    Department d ON e.departmentId = d.id
WHERE
    (e.departmentId, e.salary) IN (
        SELECT
            departmentId,
            MAX(salary)
        FROM
            Employee
        GROUP BY
            departmentId
    );
WHERE
    referee_id != 2 OR referee_id IS NULL;
