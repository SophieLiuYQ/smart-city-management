'use client';

import { useRef, useCallback, useMemo, useState } from 'react';
import Map, { Source, Layer, Popup, NavigationControl } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

import { useDashboard } from '../../context/DashboardContext';
import { ENERGY_SITES, NEXUS_SITES, scoreColor } from '../../data/sites';
import { BOROUGH_POLYGONS, EJ_POLYGONS, WASTE_DISTRICT_POINTS } from '../../data/districts';
import districtData from '../../data/top10_district_analysis.json';

/** District centroid markers from top10_district_analysis.json */
const TOP10_DISTRICTS_GEOJSON = {
  type: 'FeatureCollection',
  features: districtData.map((d, i) => ({
    type: 'Feature',
    properties: {
      rank: i + 1,
      code: d.district_code,
      borough: d.borough,
      district: d.community_district,
      solar_kwh_yr: d.buildings_summary.total_solar_potential_kwh_yr,
      bess_savings_usd: d.buildings_summary.total_bess_savings_usd_yr,
      buildings: d.buildings_summary.total,
      solar_ready: d.buildings_summary.solar_ready,
      pct_ej: d.buildings_summary.pct_ej,
    },
    geometry: { type: 'Point', coordinates: [d.centroid_lon, d.centroid_lat] },
  })),
};

/** Individual building points from each district's buildings array */
const DISTRICT_BUILDINGS_GEOJSON = {
  type: 'FeatureCollection',
  features: districtData.flatMap((d, i) =>
    d.buildings
      .filter(b => b.latitude != null && b.longitude != null)
      .map(b => ({
        type: 'Feature',
        properties: {
          districtCode: d.district_code,
          districtRank: i + 1,
          site: b.site,
          address: b.address,
          agency: b.agency,
          ej: b.ej,
          solar_kwh_yr: b.energy.solar_production_kwh_yr,
          annual_cost_usd: b.energy.est_annual_cost_usd,
          ghg_co2: b.energy.ghg_tons_co2e_yr,
          bess_kwh: b.bess_recommendation.capacity_kwh,
          bess_savings_usd: b.bess_recommendation.est_annual_savings_usd,
        },
        geometry: { type: 'Point', coordinates: [b.longitude, b.latitude] },
      }))
  ),
};
import MapLegend from './MapLegend';
import MapControls from './MapControls';

/**
 * Dark raster tile style using CARTO Dark Matter — free, no API key required.
 * All vector data layers are added on top via MapLibre Source/Layer.
 */
const DARK_MAP_STYLE = {
  version: 8,
  sources: {
    'carto-dark': {
      type: 'raster',
      tiles: ['https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'],
      tileSize: 256,
      attribution: '© CARTO © OpenStreetMap contributors',
    },
  },
  layers: [{ id: 'carto-dark-layer', type: 'raster', source: 'carto-dark' }],
};

/** Build GeoJSON FeatureCollections from the site arrays */
function buildEnergyGeoJSON(sites) {
  return {
    type: 'FeatureCollection',
    features: sites.map(s => ({
      type: 'Feature',
      properties: { ...s },
      geometry: { type: 'Point', coordinates: [s.lng, s.lat] },
    })),
  };
}

function buildNexusGeoJSON(sites) {
  return {
    type: 'FeatureCollection',
    features: sites.map(s => ({
      type: 'Feature',
      properties: { ...s },
      geometry: { type: 'Point', coordinates: [s.lng, s.lat] },
    })),
  };
}

/** Layer paint expressions */
const ENERGY_CIRCLE_LAYER = {
  id: 'energy-sites',
  type: 'circle',
  source: 'energy-sites',
  paint: {
    'circle-radius': ['interpolate', ['linear'], ['get', 'nexusScore'], 88, 7, 95, 10, 100, 13],
    'circle-color': [
      'case',
      ['>=', ['get', 'nexusScore'], 95], '#EF4444',
      ['>=', ['get', 'nexusScore'], 90], '#F59E0B',
      '#3B82F6',
    ],
    'circle-opacity': 0.9,
    'circle-stroke-color': 'rgba(255,255,255,0.15)',
    'circle-stroke-width': 1,
  },
};

const ENERGY_CIRCLE_SELECTED = {
  id: 'energy-sites-selected',
  type: 'circle',
  source: 'energy-sites',
  filter: ['==', ['get', 'id'], -1], // updated dynamically
  paint: {
    'circle-radius': ['interpolate', ['linear'], ['get', 'score'], 50, 10, 100, 16],
    'circle-color': 'transparent',
    'circle-stroke-color': '#fff',
    'circle-stroke-width': 2,
  },
};

const WASTE_FILL_LAYER = {
  id: 'borough-fill',
  type: 'fill',
  source: 'boroughs',
  paint: {
    'fill-color': [
      'interpolate', ['linear'],
      ['get', 'wasteTons'],
      28000, 'rgba(249,115,22,0.08)',
      45000, 'rgba(249,115,22,0.20)',
      65000, 'rgba(249,115,22,0.35)',
    ],
    'fill-outline-color': 'rgba(249,115,22,0.4)',
  },
};

const WASTE_CIRCLES_LAYER = {
  id: 'waste-districts',
  type: 'circle',
  source: 'waste-districts',
  paint: {
    'circle-radius': ['interpolate', ['linear'], ['get', 'ref'], 7000, 10, 12500, 20],
    'circle-color': [
      'case',
      ['>=', ['get', 'score'], 80], '#EF4444',
      ['>=', ['get', 'score'], 70], '#F59E0B',
      '#3B82F6',
    ],
    'circle-opacity': 0.85,
    'circle-stroke-color': '#0B1120',
    'circle-stroke-width': 1.5,
  },
};

const NEXUS_HEATMAP_LAYER = {
  id: 'nexus-heatmap',
  type: 'heatmap',
  source: 'nexus-sites',
  paint: {
    'heatmap-weight': ['interpolate', ['linear'], ['get', 'nScore'], 80, 0.4, 100, 1],
    'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 9, 0.8, 12, 2],
    'heatmap-color': [
      'interpolate', ['linear'], ['heatmap-density'],
      0,    'rgba(139,92,246,0)',
      0.2,  'rgba(139,92,246,0.3)',
      0.5,  'rgba(139,92,246,0.6)',
      0.8,  'rgba(168,85,247,0.85)',
      1,    'rgba(192,132,252,1)',
    ],
    'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 9, 40, 12, 80],
    'heatmap-opacity': 0.8,
  },
};

const EJ_FILL_LAYER = {
  id: 'ej-areas',
  type: 'fill',
  source: 'ej-areas',
  paint: {
    'fill-color': 'rgba(139,92,246,0.06)',
    'fill-outline-color': 'rgba(139,92,246,0.5)',
  },
};

/** Individual building dots — small circles colored by EJ status */
const BUILDING_CIRCLE_LAYER = {
  id: 'district-buildings',
  type: 'circle',
  source: 'district-buildings',
  paint: {
    'circle-radius': 5,
    'circle-color': ['case', ['==', ['get', 'ej'], true], '#10B981', '#38BDF8'],
    'circle-opacity': 0.85,
    'circle-stroke-color': '#0B1120',
    'circle-stroke-width': 1,
  },
};

/** Top-10 district circle layer — gold ring markers */
const DISTRICT_CIRCLE_LAYER = {
  id: 'top10-districts',
  type: 'circle',
  source: 'top10-districts',
  paint: {
    'circle-radius': 14,
    'circle-color': 'rgba(234,179,8,0.18)',
    'circle-stroke-color': '#EAB308',
    'circle-stroke-width': 2.5,
    'circle-opacity': 1,
  },
};

/** Rank number label on top of each district circle */
const DISTRICT_LABEL_LAYER = {
  id: 'top10-districts-labels',
  type: 'symbol',
  source: 'top10-districts',
  layout: {
    'text-field': ['to-string', ['get', 'rank']],
    'text-size': 11,
    'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'],
    'text-allow-overlap': true,
    'text-ignore-placement': true,
  },
  paint: {
    'text-color': '#FDE047',
    'text-halo-color': '#0B1120',
    'text-halo-width': 1,
  },
};

export default function CityMap() {
  const mapRef = useRef(null);
  const { viewMode, selectedId, selectSite, borough, minScore } = useDashboard();
  const [popup, setPopup] = useState(null); // { lng, lat, content }

  /** Filter energy sites by borough + minScore */
  const filteredEnergySites = useMemo(() =>
    ENERGY_SITES.filter(s =>
      (borough === 'All Boroughs' || s.borough === borough) && s.score >= minScore
    ), [borough, minScore]);

  const energyGeoJSON     = useMemo(() => buildEnergyGeoJSON(filteredEnergySites), [filteredEnergySites]);
  const nexusGeoJSON      = useMemo(() => buildNexusGeoJSON(NEXUS_SITES), []);
  const wasteDistrictData = useMemo(() => WASTE_DISTRICT_POINTS, []);

  /** FlyTo selected site */
  const flyToSite = useCallback((site) => {
    mapRef.current?.flyTo({
      center: [site.lng, site.lat],
      zoom: 14,
      duration: 900,
      essential: true,
    });
  }, []);

  const onMapClick = useCallback((e) => {
    if (!mapRef.current) return;
    const map = mapRef.current.getMap();

    // Energy mode — click on energy sites
    if (viewMode === 'energy') {
      const features = map.queryRenderedFeatures(e.point, { layers: ['energy-sites'] });
      if (features.length) {
        const props = features[0].properties;
        selectSite(props.id);
        flyToSite({ lng: props.lng, lat: props.lat });
        setPopup(null);
      } else {
        setPopup(null);
      }
    }

    // Waste mode — click on district circles
    if (viewMode === 'waste') {
      const features = map.queryRenderedFeatures(e.point, { layers: ['waste-districts'] });
      if (features.length) {
        const p = features[0].properties;
        const coords = features[0].geometry.coordinates;
        setPopup({
          lng: coords[0], lat: coords[1],
          content: (
            <div>
              <div style={{ fontWeight: 700, marginBottom: 4, color: '#F1F5F9' }}>{p.name} — {p.borough}</div>
              <div style={{ color: '#94A3B8' }}>Refuse: <span style={{ color: '#F59E0B' }}>{Number(p.ref).toLocaleString()} t/mo</span></div>
              <div style={{ color: '#94A3B8' }}>Diversion: <span style={{ color: '#10B981' }}>{p.diversion}%</span></div>
              <div style={{ color: '#94A3B8' }}>311 Complaints: <span style={{ color: '#FCA5A5' }}>{Number(p.complaints).toLocaleString()}</span></div>
              <div style={{ color: '#94A3B8' }}>Organic Potential: <span style={{ color: '#C4B5FD' }}>{p.organic}</span></div>
            </div>
          ),
        });
      }
    }

    // Nexus mode — click
    if (viewMode === 'nexus') {
      const features = map.queryRenderedFeatures(e.point, { layers: ['nexus-heatmap'] });
      if (features.length === 0) setPopup(null);
    }
  }, [viewMode, selectSite, flyToSite]);

  const onMouseMove = useCallback((e) => {
    if (!mapRef.current) return;
    const map = mapRef.current.getMap();

    // Building dots hover — checked in all view modes
    const buildingFeatures = map.queryRenderedFeatures(e.point, { layers: ['district-buildings'] });
    if (buildingFeatures.length) {
      map.getCanvas().style.cursor = 'pointer';
      const p = buildingFeatures[0].properties;
      const coords = buildingFeatures[0].geometry.coordinates;
      const fmtKwh = n => n == null ? '—' : n >= 1e6 ? `${(n/1e6).toFixed(1)}M` : n >= 1e3 ? `${(n/1e3).toFixed(0)}K` : String(Math.round(n));
      const fmtUsd = n => (!n || n === 0) ? '—' : n >= 1e6 ? `$${(n/1e6).toFixed(1)}M` : `$${(n/1e3).toFixed(0)}K`;
      setPopup({
        lng: coords[0], lat: coords[1],
        content: (
          <div>
            <div style={{ fontWeight: 700, marginBottom: 2, color: '#F1F5F9' }}>{p.site || p.address}</div>
            <div style={{ color: '#94A3B8', fontSize: 11, marginBottom: 4 }}>{p.address} · {p.agency}</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 4 }}>
              <span style={{ background: 'rgba(100,116,139,.15)', color: '#94A3B8', padding: '1px 6px', borderRadius: 20, fontSize: 10 }}>
                {p.districtCode}
              </span>
              {p.ej && <span style={{ background: 'rgba(139,92,246,.15)', color: '#C4B5FD', padding: '1px 6px', borderRadius: 20, fontSize: 10 }}>EJ</span>}
            </div>
            <div style={{ color: '#94A3B8', fontSize: 11 }}>☀ Solar: <span style={{ color: '#FDE047', fontWeight: 600 }}>{fmtKwh(p.solar_kwh_yr)} kWh/yr</span></div>
            <div style={{ color: '#94A3B8', fontSize: 11 }}>BESS: <span style={{ color: '#FCD34D' }}>{p.bess_kwh} kWh</span> · Savings: <span style={{ color: '#10B981', fontWeight: 600 }}>{fmtUsd(p.bess_savings_usd)}/yr</span></div>
            {p.ghg_co2 != null && <div style={{ color: '#94A3B8', fontSize: 11 }}>GHG: <span style={{ color: '#6EE7B7' }}>{p.ghg_co2} t CO₂/yr</span></div>}
          </div>
        ),
      });
      return;
    }

    // Top-10 district hover — checked in all view modes
    const districtFeatures = map.queryRenderedFeatures(e.point, { layers: ['top10-districts'] });
    if (districtFeatures.length) {
      map.getCanvas().style.cursor = 'pointer';
      const p = districtFeatures[0].properties;
      const coords = districtFeatures[0].geometry.coordinates;
      setPopup({
        lng: coords[0], lat: coords[1],
        content: (
          <div>
            <div style={{ fontWeight: 700, marginBottom: 4, color: 'black' }}>
              {p.address}
            </div>
            <div style={{ color: '#FDE047', marginBottom: 4, fontSize: 12 }}>
              ☀ Solar Potential: {Number(p.solar_kwh_yr).toLocaleString()} kWh/yr
            </div>
            <div style={{ color: '#94A3B8', fontSize: 11 }}>
              Buildings: <span style={{ color: '#F1F5F9' }}>{p.buildings}</span>
              {' '}· Solar-ready: <span style={{ color: '#6EE7B7' }}>{p.solar_ready}</span>
            </div>
            <div style={{ color: '#94A3B8', fontSize: 11 }}>
              BESS Savings: <span style={{ color: '#10B981' }}>${Number(p.bess_savings_usd).toLocaleString()}/yr</span>
            </div>
            {p.pct_ej > 20 && (
              <div style={{ marginTop: 4 }}>
                <span style={{ background: 'rgba(139,92,246,.15)', color: '#C4B5FD', padding: '1px 6px', borderRadius: 20, fontSize: 10 }}>
                  {p.pct_ej}% EJ
                </span>
              </div>
            )}
          </div>
        ),
      });
      return;
    }

    if (viewMode === 'energy') {
      const features = map.queryRenderedFeatures(e.point, { layers: ['energy-sites'] });
      map.getCanvas().style.cursor = features.length ? 'pointer' : '';

      if (features.length) {
        const p = features[0].properties;
        const coords = features[0].geometry.coordinates;
        setPopup({
          lng: coords[0], lat: coords[1],
          content: (
            <div>
              <div style={{ fontWeight: 700, marginBottom: 4, color: '#F1F5F9' }}>{p.name}</div>
              <div style={{ color: '#94A3B8', marginBottom: 2 }}>{p.borough} · {p.agency}</div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span style={{ color: scoreColor(p.nexusScore), fontWeight: 700, fontSize: 14 }}>{p.nexusScore}</span>
                <span style={{ color: '#64748B', fontSize: 11 }}>nexus score</span>
              </div>
              <div style={{ color: '#6EE7B7', marginTop: 2 }}>Savings: {p.savings}/yr</div>
              {p.ej && <div style={{ marginTop: 3 }}><span style={{ background: 'rgba(16,185,129,.15)', color: '#6EE7B7', padding: '1px 6px', borderRadius: 20, fontSize: 10 }}>EJ Area</span></div>}
            </div>
          ),
        });
      } else {
        setPopup(null);
      }
    } else {
      map.getCanvas().style.cursor = '';
    }
  }, [viewMode]);

  /** Selected site highlight filter */
  const selectedFilter = useMemo(() => ['==', ['get', 'id'], selectedId], [selectedId]);

  /** Geocoder search — fly to a site by name match */
  const handleSearch = useCallback((query) => {
    const q = query.toLowerCase();
    const match = ENERGY_SITES.find(s =>
      s.name.toLowerCase().includes(q) || s.borough.toLowerCase().includes(q)
    );
    if (match) {
      selectSite(match.id);
      flyToSite(match);
    }
  }, [selectSite, flyToSite]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', borderRadius: 12, overflow: 'hidden', background: '#0B1120' }}>
      <Map
        ref={mapRef}
        mapStyle={DARK_MAP_STYLE}
        initialViewState={{ longitude: -73.94, latitude: 40.71, zoom: 10.2 }}
        style={{ width: '100%', height: '100%' }}
        onClick={onMapClick}
        onMouseMove={onMouseMove}
        interactiveLayerIds={[
          'district-buildings',
          'top10-districts',
          ...(viewMode === 'energy' ? ['energy-sites'] : viewMode === 'waste' ? ['waste-districts'] : []),
        ]}
        attributionControl={false}
        logoPosition="bottom-right"
      >
        {/* Navigation controls */}
        <NavigationControl position="top-right" showCompass={false} />

        {/* ── BOROUGH FILL (Waste + Nexus modes) ── */}
        {(viewMode === 'waste' || viewMode === 'nexus') && (
          <Source id="boroughs" type="geojson" data={BOROUGH_POLYGONS}>
            <Layer {...WASTE_FILL_LAYER} />
          </Source>
        )}

        {/* ── EJ AREAS (Nexus mode only) ── */}
        {viewMode === 'nexus' && (
          <Source id="ej-areas" type="geojson" data={EJ_POLYGONS}>
            <Layer {...EJ_FILL_LAYER} />
          </Source>
        )}

        {/* ── ENERGY SITES (Energy mode) ── */}
        {viewMode === 'energy' && (
          <Source id="energy-sites" type="geojson" data={energyGeoJSON}>
            <Layer {...ENERGY_CIRCLE_LAYER} />
            <Layer
              {...ENERGY_CIRCLE_SELECTED}
              filter={selectedFilter}
              paint={{
                ...ENERGY_CIRCLE_SELECTED.paint,
                'circle-radius': 16,
              }}
            />
          </Source>
        )}

        {/* ── WASTE DISTRICT CIRCLES (Waste mode) ── */}
        {viewMode === 'waste' && (
          <Source id="waste-districts" type="geojson" data={wasteDistrictData}>
            <Layer {...WASTE_CIRCLES_LAYER} />
          </Source>
        )}

        {/* ── NEXUS HEATMAP (Nexus mode) ── */}
        {viewMode === 'nexus' && (
          <Source id="nexus-sites" type="geojson" data={nexusGeoJSON}>
            <Layer {...NEXUS_HEATMAP_LAYER} />
          </Source>
        )}

        {/* ── DISTRICT BUILDINGS (all modes) ── */}
        <Source id="district-buildings" type="geojson" data={DISTRICT_BUILDINGS_GEOJSON}>
          <Layer {...BUILDING_CIRCLE_LAYER} />
        </Source>

        {/* ── TOP-10 SOLAR DISTRICTS (all modes) ── */}
        <Source id="top10-districts" type="geojson" data={TOP10_DISTRICTS_GEOJSON}>
          <Layer {...DISTRICT_CIRCLE_LAYER} />
          <Layer {...DISTRICT_LABEL_LAYER} />
        </Source>

        {/* ── HOVER POPUP ── */}
        {popup && (
          <Popup
            longitude={popup.lng}
            latitude={popup.lat}
            closeButton={false}
            closeOnClick={false}
            anchor="bottom"
            offset={16}
          >
            {popup.content}
          </Popup>
        )}
      </Map>

      {/* Map title overlay */}
      <div style={{
        position: 'absolute', top: 10, left: 10, zIndex: 10,
        background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(4px)',
        padding: '5px 10px', borderRadius: 8,
        fontSize: 11, fontWeight: 500, color: '#CBD5E1',
        pointerEvents: 'none',
      }}>
        NYC Municipal Sites — {viewMode === 'energy' ? 'Priority Map' : viewMode === 'waste' ? 'Waste Flow' : 'Nexus Intelligence'}
      </div>

      {/* Search bar */}
      <MapControls onSearch={handleSearch} />

      {/* Legend */}
      <MapLegend />
    </div>
  );
}
