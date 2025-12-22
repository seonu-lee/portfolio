-- 17. SQL UNION
/* 1. UNION연산자: 조인을 사용하지 않고 한 데이터 세트를 다른 데이터 세트 위에 쌓을 수 있음. -> 두 개의 별도 SELECT문을 작성하고 두 명령물의 결과를 같은 테이블에 표시할 수 있음*/

SELECT *
  FROM tutorial.crunchbase_investments_part1

 UNION

 SELECT *
   FROM tutorial.crunchbase_investments_part2

/*2. union은 고유한 값만 추가함. 중복X = 추가된 테이블에서 첫번째 테이블과 동일한 행은 삭제됨
두번째 테이블의 모든 값을 추가하려면 = 중복제거X union all을 사용하면 됨 */

SELECT *
  FROM tutorial.crunchbase_investments_part1

 UNION ALL

 SELECT *
   FROM tutorial.crunchbase_investments_part2

/*3. 엄격한 규칙이 있음.  1) 두 테이블 모두 동일한 수의 열을 가져야 합니다. 2)열은 첫 번째 테이블과 동일한 순서로 동일한 데이터 유형을 가져야 합니다. 열 이름이 같을 필요는 없으나 일반적으로 같음 */

/*연습문제1: Write a query that appends the two crunchbase_investments datasets above (including duplicate values). Filter the first dataset to only companies with names that start with the letter "T", and filter the second to companies with names starting with "M" (both not case-sensitive). Only include the company_permalink, company_name, and investor_name columns (위의 두개의 crunchbase_investments 데이터셋을 중복값을 포함하는 쿼리를 작성해라. 첫번째 데이터셋은 T로 시작하는 이름을 가진 회사만 필터링, 두번째 데이터셋은 M으로 시작하는 이름을 가진 회사만 필터링. company_permalink, company_name 및 investor_name 열만 포함)*/

SELECT company_permalink,
       company_name,
       investor_name
  FROM tutorial.crunchbase_investments_part1
 WHERE company_name ILIKE 'T%'
 
 UNION ALL

SELECT company_permalink,
       company_name,
       investor_name
  FROM tutorial.crunchbase_investments_part2
 WHERE company_name ILIKE 'M%'

/*연습문제2: Write a query that shows 3 columns. The first indicates which dataset (part 1 or 2) the data comes from, the second shows company status, and the third is a count of the number of investors.
Hint: you will have to use the tutorial.crunchbase_companies table as well as the investments tables. And you'll want to group by status and dataset. (3개의 열을 표시하는 쿼리를 작성해라. 첫번째는 데이터의 출처를 나타내고 두번째는 회사상태를 나타내며 세번쨰 데이터세트는 투자자 수를 세는 것. 투자ㅔ이블 뿐만아니라 회사테이블도 사용해야한다. 상태와 데이터셋에 따라 그룹화해야함) */

SELECT 'investments_part1' AS dataset_name,
       companies.status,
       COUNT(DISTINCT investments.investor_permalink) AS investors
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_investments_part1 investments
    ON companies.permalink = investments.company_permalink
 GROUP BY 1,2

 UNION ALL
 
 SELECT 'investments_part2' AS dataset_name,
       companies.status,
       COUNT(DISTINCT investments.investor_permalink) AS investors
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_investments_part2 investments
    ON companies.permalink = investments.company_permalink
 GROUP BY 1,2

-- 18. SQL Joins with Comparison Operators

/*1. 조인과 함께 비교연산자 사용: 지금까지는 두 테이블 값을 정확히 일치시켜서 테이블을 조인했음. 조인과 함께 비교연산자를 통해 필터링을 하면 조건에 맞는 행만 조인하기 때문에 어떤 유형의 조건문이든 조인가능함 */

-- 조건에 맞는 행만 조인함
SELECT companies.permalink,
       companies.name,
       companies.status,
       COUNT(investments.investor_permalink) AS investors
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_investments_part1 investments
    ON companies.permalink = investments.company_permalink
   AND investments.funded_year > companies.founded_year + 5
 GROUP BY 1,2, 3

-- 조인이 이루어진 후 필터링됨, 위에랑 다른 결과를 생성함
SELECT companies.permalink,
       companies.name,
       companies.status,
       COUNT(investments.investor_permalink) AS investors
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_investments_part1 investments
    ON companies.permalink = investments.company_permalink
 WHERE investments.funded_year > companies.founded_year + 5
 GROUP BY 1,2, 3

--19. SQL Joins on Multiple Keys
/*1. 여러 키에 대한 조인: 여러 외래키를 기준으로 테이블을 조인함. 이유 -> 정확성, 성능(여러 필드를 조인하면 정확도가 향상되지 않더라도 속도가 향상될 수 있음)*/

--마지막줄을 추가하든 안하든 결과 같음, 하지만 추가하면 쿼리가 더 빨리 실행됨
SELECT companies.permalink,
       companies.name,
       investments.company_name,
       investments.company_permalink
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_investments_part1 investments
    ON companies.permalink = investments.company_permalink
   AND companies.name = investments.company_name

--20. SQL Self Joins
/*1. 셀프조인테이블: 자기자신의 테이블과 조인함. 서로 다른 별칭을 사용하여 동일한 테이블을 여러번 조인 가능함*/

--같은 테이블에 있는 일본투자회사와 영국투자회사를 찾고싶음
SELECT DISTINCT japan_investments.company_name,
	   japan_investments.company_permalink
  FROM tutorial.crunchbase_investments_part1 japan_investments
  JOIN tutorial.crunchbase_investments_part1 gb_investments
    ON japan_investments.company_name = gb_investments.company_name
   AND gb_investments.investor_country_code = 'GBR'
   AND gb_investments.funded_at > japan_investments.funded_at
 WHERE japan_investments.investor_country_code = 'JPN'
 ORDER BY 1