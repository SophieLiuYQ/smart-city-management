'use client';

import { createContext, useContext, useState, useCallback } from 'react';

const DashboardContext = createContext(null);

export function DashboardProvider({ children }) {
  const [viewMode, setViewMode]     = useState('energy');   // 'energy' | 'waste' | 'nexus'
  const [selectedId, setSelectedId] = useState(0);          // energy site id
  const [borough, setBorough]       = useState('All Boroughs');
  const [minScore, setMinScore]     = useState(0);
  const [budget, setBudget]         = useState(10);
  const [ejPriority, setEjPriority] = useState(true);
  const [simTab, setSimTab]         = useState('bess');      // 'bess' | 'waste' | 'scenario'
  const [selectedItem, setSelectedItem] = useState(null);   // { type: 'building'|'district'|'highlight', data: {} }
  const [flyToCoords, setFlyToCoords] = useState(null);     // { lat, lng, zoom, _t }

  // Scenario planner state
  const [scenario, setScenario] = useState({
    budget: 10, bess: 40, solar: 25, organics: 20, route: 15, equity: 0.3,
  });

  const updateScenario = useCallback((key, val) => {
    setScenario(prev => ({ ...prev, [key]: val }));
  }, []);

  const selectSite = useCallback((id) => {
    setSelectedId(id);
  }, []);

  const selectItem = useCallback((type, data) => {
    setSelectedItem({ type, data });
  }, []);

  const changeView = useCallback((v) => {
    setViewMode(v);
    if (v === 'waste')  setSimTab('waste');
    else if (v === 'nexus') setSimTab('scenario');
    else setSimTab('bess');
  }, []);

  return (
    <DashboardContext.Provider value={{
      viewMode, changeView,
      selectedId, selectSite,
      selectedItem, selectItem,
      flyToCoords, setFlyToCoords,
      borough, setBorough,
      minScore, setMinScore,
      budget, setBudget,
      ejPriority, setEjPriority,
      simTab, setSimTab,
      scenario, updateScenario,
    }}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard() {
  const ctx = useContext(DashboardContext);
  if (!ctx) throw new Error('useDashboard must be inside DashboardProvider');
  return ctx;
}
