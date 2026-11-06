SELECT
  c.name,
  COUNT(u.id) AS university_count
FROM cities AS c
JOIN universities AS u
  ON ST_Within(u.geom, c.geom)
WHERE c.admin_code LIKE '01%'
GROUP BY c.id, c.name, c.geom
ORDER BY ST_Area(c.geom::geography) DESC
LIMIT 1;
