Disaster Response Scheduling Agent
==================================

Agentic AI for disaster-response staffing built on FairLib + OpenAI. It generates schedules, checks coverage, and simulates supply usage. A simple tkinter GUI lets you run scenarios without the terminal.

Features
--------
- Agentic tool chain: `schedule_builder`, `coverage_checker`, `schedule_completeness`, `supply_monitor`.
- Deterministic scheduler: fixed horizon (default 8 hours), hourly slots, max hours/day, max consecutive slots, station + role + skills.
- Supply simulation: water, food, med kits, fuel; per-person-per-hour consumption; run-out times and remaining amounts.
- GUI: form inputs for people and supplies, one-click “Run Scheduler”, and readable summaries.

Project Structure (relevant files)
----------------------------------
- `run_midpoint_demo.py` – tools, agent builder, scheduling/supply logic, GUI entry, helpers (`run_disaster_sim`, `format_hourly_schedule`, `coverage_checker`, `schedule_completeness`, `supply_monitor` ).


Requirements & Setup
--------------------
- Python 3.x
- Packages: `fairlib`, `openai` (or compatible adapter), `python-dotenv`, `tkinter` (usually bundled)
- Install: `pip install -r requirements.txt` (if present)
- Env var: `OPENAI_API_KEY`
  - `.env` example: `OPENAI_API_KEY=your_openai_api_key_here`
  - Or set in shell: `setx OPENAI_API_KEY "your_openai_api_key_here"` (Windows) or `$env:OPENAI_API_KEY="your_openai_api_key_here"` (PowerShell session)

How the Agent Works
-------------------
1) `schedule_builder`: create schedule from people, roles/skills, stations, horizon, constraints.
2) `coverage_checker`: check required roles per slot/station.
3) `schedule_completeness`: compute covered/total hours and gap list.
4) `supply_monitor`: simulate headcount per slot, consume supplies, report remaining/run-out.

Final output (JSON keys): `coverage_rate`, `passed`, `covered_hours`, `total_hours`, `first_two_gaps`, `supplies_ok`, `hours_supported`, `supply_runout_times`, `supply_remaining`.

Running the App (GUI)
---------------------
From the project directory: `python run_midpoint_demo.py`

Inputs: total responders, water (L), food (meals), medical kits, fuel (L). Click “Run Scheduler”.

Outputs (text area):
- Supply summary: sufficiency, hours supported, remaining per supply, run-out times.
- Schedule summary: pass/fail, coverage rate, first gaps (if any).
- Hourly schedule: time range, count on shift, who/role per station.

Programmatic Use
----------------
```python
from run_midpoint_demo import run_disaster_sim

result, trace, schedule = run_disaster_sim(
    num_people=10,
    water_liters=500,
    food_meals=50,
    med_kits=50,
    fuel_liters=500,
)
print(result)    # Final JSON
print(schedule)  # Raw schedule entries
```

Assumptions / Limitations
-------------------------
- Single station; fixed role requirements.
- Responders identical (same roles/skills/station).
- Fixed horizon (e.g., 8 hours, 1-hour slots), simple rest model, max 8 hours/day.
- Static per-person-per-hour consumption rates.

Extending
---------
- Specilized jobs i.e. transport, search and rescue, medical, etc.
- Add a tool that can estimate timeline for response and build appropriate schedule
- Improved GUI to show longer schedule
