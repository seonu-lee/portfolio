-- select, from, limit

SELECT year,
       month,
       west
  FROM tutorial.us_housing_units

SELECT year,
       month,
       month_name,
       west,
       midwest,
       south,
       northeast
  FROM tutorial.us_housing_units

SELECT west AS "West Region", south AS South_Region
  FROM tutorial.us_housing_units

SELECT year AS "Year",
       month AS "Month",
       month_name AS "Month Name",
       west AS "West",
       midwest AS "Midwest",
       south AS "South",
       northeast AS "Northeast"
  FROM tutorial.us_housing_units
LIMIT 100

--where
-- comparison operators

SELECT *
  FROM tutorial.us_housing_units
 WHERE month = 1
LIMIT 100

SELECT *
  FROM tutorial.us_housing_units
 WHERE west > 30
LIMIT 100

SELECT *
  FROM tutorial.us_housing_units
 WHERE south <=20
LIMIT 100

SELECT *
  FROM tutorial.us_housing_units
 WHERE month_name != 'January'
LIMIT 100

SELECT *
  FROM tutorial.us_housing_units
 WHERE month_name > 'January'
LIMIT 100

SELECT *
  FROM tutorial.us_housing_units
 WHERE month_name > 'J'
LIMIT 100

SELECT *
  FROM tutorial.us_housing_units
 WHERE month_name ='February'
LIMIT 100

SELECT *
  FROM tutorial.us_housing_units
 WHERE month_name < 'O'
LIMIT 100

SELECT year,
       month,
       west,
       south,
       west + south AS south_plus_west
  FROM tutorial.us_housing_units
LIMIT 100

SELECT year,
       month,
       west,
       south,
       west + south - 4 * year AS nonsense_column
  FROM tutorial.us_housing_units
LIMIT 100

SELECT year,
       month,
       west,
       south,
       (west + south)/2 AS south_west_avg
  FROM tutorial.us_housing_units
LIMIT 100

SELECT *
  FROM tutorial.us_housing_units
  where (midwest+ northeast) < west
LIMIT 100

SELECT year, 
  south/ (south+west+midwest+northeast)*100 AS south_ratio,
  west/ (south+west+midwest+northeast)*100 AS west_ratio,
  midwest/ (south+west+midwest+northeast)*100 AS midwest_ratio,
  northeast/ (south+west+midwest+northeast)*100 AS northeast_ratio
  FROM tutorial.us_housing_units
  where year >= 2000
LIMIT 100

--billboard_top_100_year_end
--logical operators

SELECT *
  FROM tutorial.billboard_top_100_year_end
 ORDER BY year DESC, year_rank
LIMIT 100

--LIKE

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE "group_name" LIKE 'Snoop%'
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE group_name ILIKE 'snoop%'
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE artist ILIKE 'dr_ke'
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE "group_name" ilike '%ludacris%'
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE "group_name" like 'DJ%'
LIMIT 100

--IN

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year_rank IN (1, 2, 3)

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE group_name IN ('Elvis Presley', 'MC Hammer', 'Hammer')
LIMIT 100

--BETWEEN

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year_rank BETWEEN 5 AND 10
LIMIT 100


SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year_rank >= 5 AND year_rank <= 10
 limit 10

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year >= 1985 AND year < 1991
 order by year DESC
LIMIT 100

--IS NULL

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE artist IS NULL
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE song_name IS NULL
LIMIT 100

--AND

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2012 AND year_rank <= 10
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE group_name like '%Ludacris%' 
  and year_rank <=10
LIMIT 100


select *
  from tutorial.billboard_top_100_year_end
  where year_rank=1 and year in(1990,2000,2010)
LIMIT 100


select *
  from tutorial.billboard_top_100_year_end
  where song_name ilike '%love%' and year between 1960 and 1969
LIMIT 100

--OR

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year_rank = 5 OR artist = 'Gotye'
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2013
   AND ("group_name" ILIKE '%macklemore%' OR "group_name" ILIKE '%timberlake%')
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year_rank <= 10
   AND ("group_name" ILIKE '%Katy Perry%' OR "group_name" ILIKE '% Bon Jovi%')
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE song_name like '%California%'
   AND (year between 1970 and 1979 OR year between 1990 and 1999)
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE group_name like '%Dr. Dre%'
   AND (year < 2001 OR year > 2009)
LIMIT 100

-- NOT

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2013
   AND year_rank NOT BETWEEN 2 AND 3
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2013
   AND "group_name" NOT ILIKE '%macklemore%'
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2013
   AND artist IS NOT NULL
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2013
   AND song_name not ilike '%a%'
LIMIT 100

--ORDER BY

SELECT *
  FROM tutorial.billboard_top_100_year_end
 ORDER BY artist
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2013
 ORDER BY year_rank
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2013
 ORDER BY year_rank DESC
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2012
 ORDER BY song_name DESC
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
  WHERE year_rank <= 3
 ORDER BY year DESC, year_rank
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year_rank <= 3
 ORDER BY 2, 1 DESC
 limit 10

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year=2010
 ORDER BY year_rank, artist
LIMIT 100

--annotation

SELECT * --hi
/*ii*/
  FROM tutorial.billboard_top_100_year_end
 WHERE year=2010
 ORDER BY year_rank, artist
LIMIT 100


SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE group_name like '%T-Pain%'
 ORDER BY year_rank DESC
LIMIT 100

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year_rank between 10 and 20 and year in (1993, 2003, 2013) 
 ORDER BY year, year_rank
LIMIT 100






