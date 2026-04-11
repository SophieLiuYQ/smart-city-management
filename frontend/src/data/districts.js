/**
 * NYC Borough GeoJSON polygons — simplified approximate boundaries for MapLibre GL rendering.
 * Used as fill layers in Waste / Nexus view modes.
 */

export const BOROUGH_POLYGONS = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      properties: { name: 'Bronx', wasteTons: 48200, diversion: 15.8, sites: 892, hp: 234 },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-73.933, 40.796], [-73.933, 40.881], [-73.864, 40.881],
          [-73.785, 40.875], [-73.748, 40.830], [-73.790, 40.796],
          [-73.933, 40.796],
        ]],
      },
    },
    {
      type: 'Feature',
      properties: { name: 'Manhattan', wasteTons: 38900, diversion: 24.1, sites: 687, hp: 142 },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-74.019, 40.700], [-73.907, 40.796], [-73.907, 40.878],
          [-73.930, 40.878], [-73.930, 40.796], [-73.974, 40.796],
          [-74.019, 40.700],
        ]],
      },
    },
    {
      type: 'Feature',
      properties: { name: 'Queens', wasteTons: 55100, diversion: 20.5, sites: 1024, hp: 187 },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-73.962, 40.796], [-73.700, 40.796], [-73.700, 40.880],
          [-73.740, 40.880], [-73.785, 40.875], [-73.850, 40.796],
          [-73.962, 40.796],
        ]],
      },
    },
    {
      type: 'Feature',
      properties: { name: 'Brooklyn', wasteTons: 62400, diversion: 19.2, sites: 1105, hp: 198 },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-74.042, 40.568], [-73.832, 40.568], [-73.832, 40.700],
          [-73.959, 40.700], [-73.959, 40.796], [-74.042, 40.700],
          [-74.042, 40.568],
        ]],
      },
    },
    {
      type: 'Feature',
      properties: { name: 'Staten Island', wasteTons: 28300, diversion: 17.5, sites: 560, hp: 86 },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-74.259, 40.496], [-74.034, 40.496], [-74.034, 40.649],
          [-74.259, 40.649], [-74.259, 40.496],
        ]],
      },
    },
  ],
};

/** Approximate EJ (Environmental Justice) area polygons */
export const EJ_POLYGONS = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      properties: { name: 'South Bronx EJ', borough: 'Bronx' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-73.933, 40.796], [-73.870, 40.796], [-73.870, 40.855],
          [-73.933, 40.855], [-73.933, 40.796],
        ]],
      },
    },
    {
      type: 'Feature',
      properties: { name: 'East Harlem EJ', borough: 'Manhattan' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-73.952, 40.793], [-73.930, 40.793], [-73.930, 40.815],
          [-73.952, 40.815], [-73.952, 40.793],
        ]],
      },
    },
    {
      type: 'Feature',
      properties: { name: 'Brownsville EJ', borough: 'Brooklyn' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-73.932, 40.651], [-73.895, 40.651], [-73.895, 40.678],
          [-73.932, 40.678], [-73.932, 40.651],
        ]],
      },
    },
    {
      type: 'Feature',
      properties: { name: 'Jamaica EJ', borough: 'Queens' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [-73.820, 40.686], [-73.780, 40.686], [-73.780, 40.712],
          [-73.820, 40.712], [-73.820, 40.686],
        ]],
      },
    },
  ],
};

/** Waste district data as GeoJSON points */
export const WASTE_DISTRICT_POINTS = {
  type: 'FeatureCollection',
  features: [
    { type: 'Feature', properties: { id: 0, name: 'District 7',  borough: 'Bronx',         score: 88, ref: 12450, diversion: 14.2, complaints: 1847, organic: 'High'   }, geometry: { type: 'Point', coordinates: [-73.900, 40.838] } },
    { type: 'Feature', properties: { id: 1, name: 'District 1',  borough: 'Brooklyn',      score: 82, ref: 11820, diversion: 16.8, complaints: 1623, organic: 'High'   }, geometry: { type: 'Point', coordinates: [-73.985, 40.695] } },
    { type: 'Feature', properties: { id: 2, name: 'District 12', borough: 'Queens',        score: 79, ref: 10950, diversion: 18.1, complaints: 1204, organic: 'Medium' }, geometry: { type: 'Point', coordinates: [-73.825, 40.755] } },
    { type: 'Feature', properties: { id: 3, name: 'District 9',  borough: 'Manhattan',     score: 76, ref:  9870, diversion: 22.4, complaints: 2156, organic: 'Medium' }, geometry: { type: 'Point', coordinates: [-73.945, 40.805] } },
    { type: 'Feature', properties: { id: 4, name: 'District 3',  borough: 'Bronx',         score: 85, ref:  9540, diversion: 13.7, complaints: 1932, organic: 'High'   }, geometry: { type: 'Point', coordinates: [-73.915, 40.825] } },
    { type: 'Feature', properties: { id: 5, name: 'District 14', borough: 'Brooklyn',      score: 73, ref:  9120, diversion: 19.3, complaints:  987, organic: 'Medium' }, geometry: { type: 'Point', coordinates: [-73.960, 40.638] } },
    { type: 'Feature', properties: { id: 6, name: 'District 7',  borough: 'Queens',        score: 70, ref:  8760, diversion: 20.1, complaints:  876, organic: 'Medium' }, geometry: { type: 'Point', coordinates: [-73.792, 40.710] } },
    { type: 'Feature', properties: { id: 7, name: 'District 1',  borough: 'Staten Island', score: 68, ref:  8340, diversion: 17.5, complaints:  654, organic: 'Low'    }, geometry: { type: 'Point', coordinates: [-74.060, 40.645] } },
    { type: 'Feature', properties: { id: 8, name: 'District 4',  borough: 'Manhattan',     score: 64, ref:  7890, diversion: 25.6, complaints: 1543, organic: 'Low'    }, geometry: { type: 'Point', coordinates: [-73.978, 40.745] } },
    { type: 'Feature', properties: { id: 9, name: 'District 5',  borough: 'Brooklyn',      score: 61, ref:  7650, diversion: 21.2, complaints: 1098, organic: 'Medium' }, geometry: { type: 'Point', coordinates: [-73.944, 40.658] } },
  ],
};
