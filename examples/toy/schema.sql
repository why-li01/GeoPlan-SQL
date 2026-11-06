-- Minimal synthetic schema for the toy example (EPSG:4326).
CREATE EXTENSION IF NOT EXISTS postgis;

DROP TABLE IF EXISTS universities;
DROP TABLE IF EXISTS cities;

CREATE TABLE cities (
    id   integer PRIMARY KEY,
    name text NOT NULL,
    admin_code text NOT NULL,  -- e.g. region code prefix
    geom geometry(MultiPolygon, 4326) NOT NULL
);

CREATE TABLE universities (
    id   integer PRIMARY KEY,
    name text NOT NULL,
    geom geometry(Point, 4326) NOT NULL
);

INSERT INTO cities (id, name, admin_code, geom) VALUES
(
  1,
  'Riverdale',
  '01',
  ST_GeomFromText(
    'MULTIPOLYGON(((10.0 10.0, 11.0 10.0, 11.0 11.0, 10.0 11.0, 10.0 10.0)))',
    4326
  )
),
(
  2,
  'Lakeside',
  '01',
  ST_GeomFromText(
    'MULTIPOLYGON(((9.0 9.5, 9.5 9.5, 9.5 10.2, 9.0 10.2, 9.0 9.5)))',
    4326
  )
);

INSERT INTO universities (id, name, geom) VALUES
(1, 'Riverdale U', ST_GeomFromText('POINT(10.3 10.5)', 4326)),
(2, 'East Campus', ST_GeomFromText('POINT(10.7 10.4)', 4326)),
(3, 'Lakeside College', ST_GeomFromText('POINT(9.2 9.8)', 4326));
