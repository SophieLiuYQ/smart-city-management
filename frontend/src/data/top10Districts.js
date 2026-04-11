/**
 * Top 10 NYC community districts by total solar potential (kWh/yr)
 * Derived from data/gold/district_analysis.json — centroid coordinates are real.
 */
export const TOP10_DISTRICTS_GEOJSON = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      properties: {
        rank: 1, code: 'MN11', borough: 'Manhattan', district: 11,
        solar_kwh_yr: 9469229, bess_savings_usd: 7287775,
        buildings: 168, solar_ready: 13, pct_ej: 47.0,
      },
      geometry: { type: 'Point', coordinates: [-73.933877, 40.793766] },
    },
    {
      type: 'Feature',
      properties: {
        rank: 2, code: 'SI03', borough: 'Staten Island', district: 3,
        solar_kwh_yr: 3877599, bess_savings_usd: 4657250,
        buildings: 66, solar_ready: 12, pct_ej: 1.5,
      },
      geometry: { type: 'Point', coordinates: [-74.178621, 40.545108] },
    },
    {
      type: 'Feature',
      properties: {
        rank: 3, code: 'BK18', borough: 'Brooklyn', district: 18,
        solar_kwh_yr: 3142007, bess_savings_usd: 4604310,
        buildings: 74, solar_ready: 17, pct_ej: 33.8,
      },
      geometry: { type: 'Point', coordinates: [-73.914608, 40.633022] },
    },
    {
      type: 'Feature',
      properties: {
        rank: 4, code: 'BK03', borough: 'Brooklyn', district: 3,
        solar_kwh_yr: 2752211, bess_savings_usd: 5820490,
        buildings: 85, solar_ready: 12, pct_ej: 47.1,
      },
      geometry: { type: 'Point', coordinates: [-73.941837, 40.68837] },
    },
    {
      type: 'Feature',
      properties: {
        rank: 5, code: 'BK15', borough: 'Brooklyn', district: 15,
        solar_kwh_yr: 2684451, bess_savings_usd: 4798185,
        buildings: 104, solar_ready: 16, pct_ej: 2.9,
      },
      geometry: { type: 'Point', coordinates: [-73.942234, 40.58888] },
    },
    {
      type: 'Feature',
      properties: {
        rank: 6, code: 'SI02', borough: 'Staten Island', district: 2,
        solar_kwh_yr: 2561966, bess_savings_usd: 3509750,
        buildings: 62, solar_ready: 7, pct_ej: 11.3,
      },
      geometry: { type: 'Point', coordinates: [-74.124792, 40.588701] },
    },
    {
      type: 'Feature',
      properties: {
        rank: 7, code: 'QN11', borough: 'Queens', district: 11,
        solar_kwh_yr: 2473691, bess_savings_usd: 4451215,
        buildings: 82, solar_ready: 14, pct_ej: 29.3,
      },
      geometry: { type: 'Point', coordinates: [-73.7592, 40.755966] },
    },
    {
      type: 'Feature',
      properties: {
        rank: 8, code: 'SI01', borough: 'Staten Island', district: 1,
        solar_kwh_yr: 2442055, bess_savings_usd: 6616015,
        buildings: 157, solar_ready: 14, pct_ej: 24.8,
      },
      geometry: { type: 'Point', coordinates: [-74.103173, 40.635159] },
    },
    {
      type: 'Feature',
      properties: {
        rank: 9, code: 'QN12', borough: 'Queens', district: 12,
        solar_kwh_yr: 2177351, bess_savings_usd: 6897165,
        buildings: 107, solar_ready: 12, pct_ej: 49.5,
      },
      geometry: { type: 'Point', coordinates: [-73.784216, 40.695422] },
    },
    {
      type: 'Feature',
      properties: {
        rank: 10, code: 'BX09', borough: 'Bronx', district: 9,
        solar_kwh_yr: 2100417, bess_savings_usd: 4434760,
        buildings: 58, solar_ready: 11, pct_ej: 41.4,
      },
      geometry: { type: 'Point', coordinates: [-73.863099, 40.826027] },
    },
  ],
};
