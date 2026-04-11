/**
 * Chart data extracted from City_Planning_Nexus.html
 * Used by Simulation and BoroughComparison components.
 */

export const HOURLY_DISPATCH = [
  { hour: 0,  label: "12AM", demand: 62,  solar: 0,   bess: 0   },
  { hour: 1,  label: "1AM",  demand: 58,  solar: 0,   bess: 0   },
  { hour: 2,  label: "2AM",  demand: 52,  solar: 0,   bess: 0   },
  { hour: 3,  label: "3AM",  demand: 50,  solar: 0,   bess: 0   },
  { hour: 4,  label: "4AM",  demand: 52,  solar: 0,   bess: 0   },
  { hour: 5,  label: "5AM",  demand: 60,  solar: 2,   bess: 0   },
  { hour: 6,  label: "6AM",  demand: 95,  solar: 10,  bess: 0   },
  { hour: 7,  label: "7AM",  demand: 140, solar: 30,  bess: 15  },
  { hour: 8,  label: "8AM",  demand: 185, solar: 70,  bess: 45  },
  { hour: 9,  label: "9AM",  demand: 220, solar: 120, bess: 80  },
  { hour: 10, label: "10AM", demand: 260, solar: 160, bess: 105 },
  { hour: 11, label: "11AM", demand: 285, solar: 190, bess: 120 },
  { hour: 12, label: "12PM", demand: 295, solar: 200, bess: 125 },
  { hour: 13, label: "1PM",  demand: 290, solar: 190, bess: 120 },
  { hour: 14, label: "2PM",  demand: 280, solar: 170, bess: 110 },
  { hour: 15, label: "3PM",  demand: 265, solar: 140, bess: 95  },
  { hour: 16, label: "4PM",  demand: 240, solar: 100, bess: 70  },
  { hour: 17, label: "5PM",  demand: 200, solar: 50,  bess: 40  },
  { hour: 18, label: "6PM",  demand: 160, solar: 15,  bess: 15  },
  { hour: 19, label: "7PM",  demand: 130, solar: 0,   bess: 0   },
  { hour: 20, label: "8PM",  demand: 105, solar: 0,   bess: 0   },
  { hour: 21, label: "9PM",  demand: 88,  solar: 0,   bess: 0   },
  { hour: 22, label: "10PM", demand: 75,  solar: 0,   bess: 0   },
  { hour: 23, label: "11PM", demand: 68,  solar: 0,   bess: 0   },
];

export const SOC_DATA = [
  { hour: 0,  soc: 20 }, { hour: 1,  soc: 32 }, { hour: 2,  soc: 45 },
  { hour: 3,  soc: 58 }, { hour: 4,  soc: 72 }, { hour: 5,  soc: 85 },
  { hour: 6,  soc: 92 }, { hour: 7,  soc: 88 }, { hour: 8,  soc: 78 },
  { hour: 9,  soc: 65 }, { hour: 10, soc: 52 }, { hour: 11, soc: 40 },
  { hour: 12, soc: 30 }, { hour: 13, soc: 25 }, { hour: 14, soc: 22 },
  { hour: 15, soc: 18 }, { hour: 16, soc: 15 }, { hour: 17, soc: 14 },
  { hour: 18, soc: 16 }, { hour: 19, soc: 20 }, { hour: 20, soc: 22 },
  { hour: 21, soc: 18 }, { hour: 22, soc: 15 }, { hour: 23, soc: 18 },
];

export const WASTE_FORECAST = [
  { month: "May", refuse: 285, recycling: 52, organics: 15, total: 352 },
  { month: "Jun", refuse: 278, recycling: 54, organics: 18, total: 350 },
  { month: "Jul", refuse: 290, recycling: 53, organics: 20, total: 363 },
  { month: "Aug", refuse: 282, recycling: 55, organics: 22, total: 359 },
  { month: "Sep", refuse: 275, recycling: 56, organics: 25, total: 356 },
  { month: "Oct", refuse: 268, recycling: 58, organics: 28, total: 354 },
  { month: "Nov", refuse: 260, recycling: 59, organics: 31, total: 350 },
  { month: "Dec", refuse: 255, recycling: 60, organics: 33, total: 348 },
  { month: "Jan", refuse: 248, recycling: 62, organics: 36, total: 346 },
  { month: "Feb", refuse: 242, recycling: 63, organics: 38, total: 343 },
  { month: "Mar", refuse: 235, recycling: 65, organics: 41, total: 341 },
  { month: "Apr", refuse: 230, recycling: 66, organics: 44, total: 340 },
];

export const DIVERSION_GAP = [
  { material: "Organics", current: 14, target: 45, gap: 31 },
  { material: "Paper",    current: 52, target: 70, gap: 18 },
  { material: "Plastic",  current: 18, target: 40, gap: 22 },
  { material: "Metal",    current: 65, target: 80, gap: 15 },
  { material: "Glass",    current: 42, target: 65, gap: 23 },
];

export const SCENARIO_DEPLOYMENTS = [
  { priority: 1, action: "BESS 750kWh",  location: "PS 123 School, Bronx",         invest: "$340K", ret: "$287K/yr", co2: "42t", payback: "1.2yr", ej: true  },
  { priority: 2, action: "Solar 200kW",  location: "Navy Yard Bldg 77, Brooklyn",  invest: "$480K", ret: "$198K/yr", co2: "67t", payback: "2.4yr", ej: false },
  { priority: 3, action: "Organics AD",  location: "District 7, Bronx",            invest: "$220K", ret: "$165K/yr", co2: "38t", payback: "1.3yr", ej: true  },
  { priority: 4, action: "BESS 500kWh",  location: "FDNY Engine 42, Bronx",        invest: "$290K", ret: "$243K/yr", co2: "35t", payback: "1.2yr", ej: true  },
  { priority: 5, action: "Solar 150kW",  location: "Queens Central Lib",           invest: "$360K", ret: "$156K/yr", co2: "52t", payback: "2.3yr", ej: true  },
  { priority: 6, action: "Route Opt",    location: "Districts 3+7, Bronx",         invest: "$180K", ret: "$142K/yr", co2: "28t", payback: "1.3yr", ej: true  },
  { priority: 7, action: "BESS 600kWh",  location: "Bronx Courthouse",             invest: "$320K", ret: "$265K/yr", co2: "39t", payback: "1.2yr", ej: true  },
  { priority: 8, action: "Organics AD",  location: "District 1, Brooklyn",         invest: "$195K", ret: "$134K/yr", co2: "31t", payback: "1.5yr", ej: false },
];
