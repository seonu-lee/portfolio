--13. SQL LEFT JOIN

SELECT companies.permalink AS companies_permalink,
       companies.name AS companies_name,
       acquisitions.company_permalink AS acquisitions_permalink,
       acquisitions.acquired_at AS acquired_date
  FROM tutorial.crunchbase_companies companies
  JOIN tutorial.crunchbase_acquisitions acquisitions
    ON companies.permalink = acquisitions.company_permalink
--companies 테이블과 동일한 매핑값이 acquisitions 테이블에 2개있으면 두 번 매핑되어 두 행으로 나타남 

SELECT companies.permalink AS companies_permalink,
       companies.name AS companies_name,
       acquisitions.company_permalink AS acquisitions_permalink,
       acquisitions.acquired_at AS acquired_date
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_acquisitions acquisitions
    ON companies.permalink = acquisitions.company_permalink
-- 테이블에 일치하는 항목이 있는지 여부에 관계없이 LEFT JOIN테이블에 있는 모든 행을 반환하게 함

--연습문제: Write a query that performs an inner join between the tutorial.crunchbase_acquisitions table and the tutorial.crunchbase_companies table, but instead of listing individual rows, count the number of non-null rows in each table.
SELECT COUNT(companies.permalink) AS companies_rowcount, --1673
       COUNT(acquisitions.company_permalink) AS acquisitions_rowcount --1673
  FROM tutorial.crunchbase_companies companies
  JOIN tutorial.crunchbase_acquisitions acquisitions
    ON companies.permalink = acquisitions.company_permalink
--inner join은 두 테이블 모두 null값을 포함하고 있지 않아 행의 수가 같음

--연습문제: Modify the query above to be a LEFT JOIN. Note the difference in results.
SELECT COUNT(companies.permalink) AS companies_rowcount, --27355
       COUNT(acquisitions.company_permalink) AS acquisitions_rowcount --1673
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_acquisitions acquisitions
    ON companies.permalink = acquisitions.company_permalink
-- count()는 null을 제외하고 셈,  LEFT JOIN테이블은 null값이 존재하지 않고, null값을 가진 acquisitions 테이블의 행의 수가 더 적음

--연습문제: Count the number of unique companies (don't double-count companies) and unique acquired companies by state. Do not include results for which there is no state data, and order by the number of acquired companies from highest to lowest.
SELECT companies.state_code,
       COUNT(DISTINCT companies.permalink) AS unique_companies,
       COUNT(DISTINCT acquisitions.company_permalink) AS unique_companies_acquired
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_acquisitions acquisitions
    ON companies.permalink = acquisitions.company_permalink
 WHERE companies.state_code IS NOT NULL
 GROUP BY 1
 ORDER BY 3 DESC

-- 14. SQL RIGHT JOIN
/* left join에서 두 테이블의 이름을 바꾸면 right join과 같음 */ 

--연습문제: Rewrite the previous practice query in which you counted total and acquired companies by state, but with a RIGHT JOIN instead of a LEFT JOIN. The goal is to produce the exact same results.
SELECT companies.state_code,
       COUNT(DISTINCT companies.permalink) AS unique_companies,
       COUNT(DISTINCT acquisitions.company_permalink) AS acquired_companies
  FROM tutorial.crunchbase_acquisitions acquisitions
 RIGHT JOIN tutorial.crunchbase_companies companies
    ON companies.permalink = acquisitions.company_permalink
 WHERE companies.state_code IS NOT NULL
 GROUP BY 1
 ORDER BY 3 DESC

-- 15. SQL Joins Using WHERE or ON
/* 일반적으로는 두 테이블이 이미 조인된 후에 WHERE절에서 필터링 됨.
1. ON 절에서 필터링: 하지만 조인 전에 필터링 해야될 수도 있음 
2. WHERE 절에서 필터링: 조인된 후에 필터링 됨 */ 

SELECT companies.permalink AS companies_permalink,
       companies.name AS companies_name,
       acquisitions.company_permalink AS acquisitions_permalink,
       acquisitions.acquired_at AS acquired_date
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_acquisitions acquisitions
    ON companies.permalink = acquisitions.company_permalink
   AND acquisitions.company_permalink != '/company/1000memories'
 ORDER BY 1
-- 조인 발생 전에 조건문 AND가 평가됨, 이는 테이블 중 해당 테이블 하나에만 적용됨, 다른 테이블에서는 여전히 표시될 수 있음

SELECT companies.permalink AS companies_permalink,
       companies.name AS companies_name,
       acquisitions.company_permalink AS acquisitions_permalink,
       acquisitions.acquired_at AS acquired_date
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_acquisitions acquisitions
    ON companies.permalink = acquisitions.company_permalink
 WHERE acquisitions.company_permalink != '/company/1000memories'
    OR acquisitions.company_permalink IS NULL --null값도 필터링 될 수 있으므로 null값을 포함하도록 추가함
 ORDER BY 1
-- 조인을 한 뒤에 필터링 했기 때문에 acquisitions에 1000memories이 포함된 행이 아예 반환되지 않음

--연습문제: Write a query that shows a company's name, "status" (found in the Companies table), and the number of unique investors in that company. Order by the number of investors from most to fewest. Limit to only companies in the state of New York.
SELECT companies.name AS company_name, 
       companies.status, 
       COUNT(DISTINCT investments.investor_name) AS unqiue_investors --해당 회사의 고유 투자자 수
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_investments investments
    ON companies.permalink = investments.company_permalink
 WHERE companies.state_code = 'NY' --뉴욕주에 있는 회사만
 GROUP BY 1,2
 ORDER BY 3 DESC --투자자 수가 많은 순서대로 정렬

--연습문제: Write a query that lists investors based on the number of companies in which they are invested. Include a row for companies with no investor, and order from most companies to least.
SELECT CASE WHEN investments.investor_name IS NULL THEN 'No Investors' --투자자가 없는 회사 행을 추가
            ELSE investments.investor_name END AS investor,
       COUNT(DISTINCT companies.permalink) AS companies_invested_in
  FROM tutorial.crunchbase_companies companies
  LEFT JOIN tutorial.crunchbase_investments investments
    ON companies.permalink = investments.company_permalink
 GROUP BY 1 --투자자가 투자한 회사 수를 기준
 ORDER BY 2 DESC --투자자가 많은 회사부터 적은 회사 순으로 정렬

-- 16. SQL FULL OUTER JOIN
/* 1. (= FULL JOIN) 일반적으로 집계와 함께 사용되어 두 테이블 간의 겹치는 정도를 파악하는 데 사용됨*/

SELECT COUNT(CASE WHEN companies.permalink IS NOT NULL AND acquisitions.company_permalink IS NULL
                  THEN companies.permalink ELSE NULL END) AS companies_only,
       COUNT(CASE WHEN companies.permalink IS NOT NULL AND acquisitions.company_permalink IS NOT NULL
                  THEN companies.permalink ELSE NULL END) AS both_tables,
       COUNT(CASE WHEN companies.permalink IS NULL AND acquisitions.company_permalink IS NOT NULL
                  THEN acquisitions.company_permalink ELSE NULL END) AS acquisitions_only
  FROM tutorial.crunchbase_companies companies
  FULL JOIN tutorial.crunchbase_acquisitions acquisitions
    ON companies.permalink = acquisitions.company_permalink

--연습문제: Write a query that joins tutorial.crunchbase_companies and tutorial.crunchbase_investments_part1 using a FULL JOIN. Count up the number of rows that are matched/unmatched as in the example above.
SELECT COUNT(CASE WHEN companies.permalink IS NOT NULL AND investments.company_permalink IS NULL
                      THEN companies.permalink ELSE NULL END) AS companies_only,
           COUNT(CASE WHEN companies.permalink IS NOT NULL AND investments.company_permalink IS NOT NULL
                      THEN companies.permalink ELSE NULL END) AS both_tables,
           COUNT(CASE WHEN companies.permalink IS NULL AND investments.company_permalink IS NOT NULL
                      THEN investments.company_permalink ELSE NULL END) AS investments_only
      FROM tutorial.crunchbase_companies companies
      FULL JOIN tutorial.crunchbase_investments_part1 investments
        ON companies.permalink = investments.company_permalink
