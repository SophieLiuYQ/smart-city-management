/home/acergn100_6/run_xgboost_with_monitoring.sh: line 35: python: command not found

## Post-run GPU State
name, memory.total [MiB], driver_version, temperature.gpu, utilization.gpu [%], clocks.current.sm [MHz], clocks.current.memory [MHz]
NVIDIA GB10, [N/A], 580.142, 42, 0 %, 2405 MHz, [N/A]

## Post-run CPU Thermal
acpitz: 52C
acpitz: 43C
acpitz: 52C
acpitz: 43C
acpitz: 43C
acpitz: 43C
acpitz: 43C

## GPU Utilization During Run (sampled every 2s)
Timestamp,Temp(C),GPU_Util(%),Mem_Util(%),Clock(MHz)
42,0,0,2405

## Summary
Total runtime: 0s
Peak GPU temp: 42C
Peak GPU util: 0%
[XGBoost] v3.2.0
=================================================================
  XGBoost END-TO-END: Raw CSVs → Scores
  Raw data: /home/acergn100_6/smart-city-management/data/raw
=================================================================

── PHASE 1: LOAD RAW DATA ──
  [E5 Solar Readiness] → 4268 sites
(0.02s)
  [E3 Electric Consumption] → 147 developments
(1.90s)
  [E4 EV Stations] → 1322 stations
(0.00s)
  [E7 LL84 Monthly (2.2M rows)] → 46543 properties
(4.96s)
  [E10 Benchmarking] → 44613 properties
(1.13s)
  [W1 Tonnage] → 59 districts
(0.01s)
  [W2 311 Complaints] → 77 districts
(11.33s)
  [W7 Compost] → 591 sites
(0.00s)
  [W8 Transfer Stations] → 19 NYC stations
(0.00s)

── BUILDING FEATURE MATRIX ──
  [BBL joins (E10)] → 0 matched
(0.01s)
  [E7 energy join via BBL] → 3133 matched
(0.03s)
  [District joins (W1+W2)] → 0 with waste data
(0.00s)
  [Spatial: EV stations] → done
(0.06s)
  [Spatial: Composting] → done
(0.02s)
  [Spatial: Transfer stations] → done
(0.00s)

  Feature matrix: 4268 sites × 31 features

── PHASE 4: TRAIN & SCORE ──
  [XGBoost] GPU mode
    energy_score: 0.46s  mean=44.2  range=[16, 99]
    waste_score: 0.23s  mean=56.8  range=[31, 99]
    nexus_score: 0.23s  mean=43.3  range=[19, 99]

── PHASE 5: SAVE RESULTS ──

[SAVED] /home/acergn100_6/smart-city-management/data/gold/ranked_sites_xgboost_raw.parquet — 4268 sites
[SAVED] /home/acergn100_6/smart-city-management/data/gold/top50_xgboost_raw.json

[IMPORTANCE] Top 10 features for nexus_score:
    is_ej                           gain: 18252.2
    solar_kwh                       gain: 1763.8
    e7_avg_total                    gain: 285.9
    roof_good                       gain: 237.9
    ghg                             gain: 164.4
    solar_savings                   gain: 154.5
    sqft                            gain: 153.4
    ev_1km                          gain: 150.7
    boro_bronx                      gain: 146.3
    nearest_compost_m               gain: 88.2

  TOP 10 (by Nexus Score):
  ─────────────────────────────────────────────────────────────────
  #  1  E: 72  W: 95  N: 99  Staten Island  Ocean Breeze Indoor Athletic Facili
  #  2  E: 90  W: 85  N: 99         Bronx  Public Safety Answering Center II
  #  3  E: 87  W: 92  N: 99      Brooklyn  K276
  #  4  E: 79  W: 75  N: 99     Manhattan  Life Family Shelter
  #  5  E: 78  W: 95  N: 99        Queens  Q042
  #  6  E: 89  W: 76  N: 99     Manhattan  M485
  #  7  E: 99  W: 70  N: 99     Manhattan  Henry J. Carter Hospital
  #  8  E: 78  W: 88  N: 98      Brooklyn  K422
  #  9  E: 88  W: 79  N: 98         Bronx  X036
  # 10  E: 77  W: 81  N: 97     Manhattan  M520
  ─────────────────────────────────────────────────────────────────

=================================================================
  COMPLETE — 21.0s total
    Data loading:   20.0s
    Training:       0.9s (3 models)
    Sites scored:   4268
    Throughput:     204 sites/sec
    Features:       31
=================================================================

## Post-run GPU State
name, memory.total [MiB], driver_version, temperature.gpu, utilization.gpu [%], clocks.current.sm [MHz], clocks.current.memory [MHz]
NVIDIA GB10, [N/A], 580.142, 44, 66 %, 2528 MHz, [N/A]

## Post-run CPU Thermal
acpitz: 57C
acpitz: 46C
acpitz: 57C
acpitz: 45C
acpitz: 47C
acpitz: 46C
acpitz: 48C

## GPU Utilization During Run (sampled every 2s)
Timestamp,Temp(C),GPU_Util(%),Mem_Util(%),Clock(MHz)
42,0,0,2405
42,0,0,2405
43,0,0,2405
43,0,0,2405
43,0,0,2405
43,0,0,2405
43,0,0,2405
43,0,0,2405
43,0,0,2405
44,0,0,2405
44,0,0,2405

## Summary
Total runtime: 21s
Peak GPU temp: 44C
Peak GPU util: 0%
