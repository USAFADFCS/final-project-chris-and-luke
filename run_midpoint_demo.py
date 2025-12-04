import os, json, asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText


from fairlib import (
    SimpleAgent,
    ToolRegistry,
    ToolExecutor,
    ReActPlanner,
    OpenAIAdapter,
    WorkingMemory,
    RoleDefinition,
)


from project_tools.coverage_checker_tool import CoverageCheckerTool
from project_tools.schedule_completeness_tool import ScheduleCompletenessTool


# --- ENV SETUP ---
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY missing — set it in .env or $env:OPENAI_API_KEY")


# --- SMART SCHEDULE BUILDER (FUNCTION) ---
def build_schedule_from_scenario(scenario: dict, rules: dict):
    """
    Smart-ish schedule builder:
    - Splits the horizon into time slots.
    - For each slot and required role at each station, assigns a person:
        - who has the required role,
        - has the required skills,
        - is assigned to the right station,
        - has not exceeded max_hours_per_day,
        - is not over a max_consecutive_slots threshold (simple rest behavior).
    - If no suitable person is found, that role/slot remains unfilled (will show as a gap).

    Returns a list of schedule entries in the same format the tools expect.
    """

    horizon_start = scenario["horizon_start"]
    horizon_end = scenario["horizon_end"]
    slot_minutes = scenario["slot_minutes"]

    start_dt = datetime.fromisoformat(horizon_start)
    end_dt = datetime.fromisoformat(horizon_end)
    slot_delta = timedelta(minutes=slot_minutes)

    # Build the list of time slots
    slots = []
    curr = start_dt
    slot_index = 0
    while curr < end_dt:
        slots.append((slot_index, curr, curr + slot_delta))
        curr += slot_delta
        slot_index += 1

    # Default constraints (could later be moved into scenario["constraints"])
    max_hours_per_day = 8.0
    max_consecutive_slots = 4  # e.g., after 4 consecutive slots (4 hours), force a break
    hours_per_slot = slot_minutes / 60.0

    # Initialize per-person state
    person_state = {}
    for person in scenario["people"]:
        pid = person["person_id"]
        person_state[pid] = {
            "info": person,
            "assigned_hours": 0.0,
            "last_slot_index": None,
            "consecutive_slots": 0,
        }

    # We'll rotate through candidates per (station, role) to keep it fair-ish
    role_pointers = {}  # key: (station_id, role) -> int

    schedule = []

    stations = rules.get("stations", [])
    for (s_idx, slot_start, slot_end) in slots:
        # Track who is already scheduled in this slot to avoid double-assigning
        already_scheduled_this_slot = set()

        for station in stations:
            station_id = station["station_id"]
            required_roles = station.get("required", [])
            min_skill_per_role = station.get("min_skill_per_role", {})

            for req in required_roles:
                role = req["role"]
                count = req["count"]
                key = (station_id, role)

                # Gather candidates who can do this role at this station
                candidates = []
                for pid, state in person_state.items():
                    person = state["info"]

                    # Must match station
                    if person.get("station_id") != station_id:
                        continue

                    # Must have this role
                    if role not in person.get("roles", []):
                        continue

                    # Must have required skills for this role (if specified)
                    required_skills = min_skill_per_role.get(role, [])
                    if not all(skill in person.get("skills", []) for skill in required_skills):
                        continue

                    # Check hour limit
                    if state["assigned_hours"] + hours_per_slot > max_hours_per_day:
                        continue

                    # Simple rest: limit consecutive slots
                    last_idx = state["last_slot_index"]
                    consec = state["consecutive_slots"]
                    if last_idx is not None and s_idx == last_idx + 1 and consec >= max_consecutive_slots:
                        # Needs a break this slot
                        continue

                    candidates.append(pid)

                if not candidates:
                    # No one available; this will be a gap for this role/slot
                    continue

                # Round-robin through candidates to spread load
                start_pointer = role_pointers.get(key, 0)

                needed = count
                attempts = 0
                assigned_this_role = 0
                num_candidates = len(candidates)

                while needed > 0 and attempts < num_candidates:
                    idx = (start_pointer + attempts) % num_candidates
                    pid = candidates[idx]
                    if pid in already_scheduled_this_slot:
                        attempts += 1
                        continue

                    # Assign this person for the slot
                    state = person_state[pid]
                    # Update state
                    if state["last_slot_index"] is not None and s_idx == state["last_slot_index"] + 1:
                        consec = state["consecutive_slots"] + 1
                    else:
                        consec = 1

                    state["last_slot_index"] = s_idx
                    state["consecutive_slots"] = consec
                    state["assigned_hours"] += hours_per_slot

                    # Add to schedule
                    info = state["info"]
                    schedule.append(
                        {
                            "person_id": pid,
                            "person_roles": [role],
                            "person_skills": info["skills"],
                            "station_id": info["station_id"],
                            "start": slot_start.isoformat(),
                            "end": slot_end.isoformat(),
                        }
                    )

                    already_scheduled_this_slot.add(pid)
                    assigned_this_role += 1
                    needed -= 1
                    attempts += 1

                # Update pointer for next time
                role_pointers[key] = (start_pointer + assigned_this_role) % num_candidates

    return schedule


# --- SUPPLY SIM (FUNCTION) ---
def simulate_supplies(scenario: dict, schedule: list):
    """
    Simple supply simulation:
    - Counts how many unique people are working in each time slot.
    - Applies fixed per-person-per-hour consumption rates for each supply.
    - Computes total usage, remaining amounts, whether supplies last,
      and the time when each supply runs out (if they do).

    Returns a dict with:
      {
        "initial_supplies": {...},
        "total_usage": {...},
        "remaining": {...},
        "supplies_ok": bool,
        "runout_times": {supply_name: ISO datetime or None},
        "hours_supported": float
      }
    """

    horizon_start = scenario["horizon_start"]
    horizon_end = scenario["horizon_end"]
    slot_minutes = scenario["slot_minutes"]
    supplies = scenario.get("supplies", {})

    if not supplies:
        return {
            "initial_supplies": {},
            "total_usage": {},
            "remaining": {},
            "supplies_ok": True,
            "runout_times": {},
            "hours_supported": 0.0,
        }

    start_dt = datetime.fromisoformat(horizon_start)
    end_dt = datetime.fromisoformat(horizon_end)
    slot_delta = timedelta(minutes=slot_minutes)
    hours_per_slot = slot_minutes / 60.0

    # Default per-person-per-hour consumption rates
    default_rates = {
        "water_liters": 2.0,  # liters per person per hour
        "food_meals": 0.5,    # meals per person per hour
        "med_kits": 0.05,     # kits per person per hour
        "fuel_liters": 0.5,   # liters per person per hour
    }

    # Allow overrides via scenario["consumption_rates"]
    rates = default_rates.copy()
    rates.update(scenario.get("consumption_rates", {}))

    # Initialize trackers
    total_usage = {name: 0.0 for name in supplies.keys()}
    runout_times = {name: None for name in supplies.keys()}
    supplies_ok = True

    # Helper: check if a schedule entry overlaps a given slot
    def overlaps(entry, slot_start, slot_end):
        es = datetime.fromisoformat(entry["start"])
        ee = datetime.fromisoformat(entry["end"])
        return not (ee <= slot_start or es >= slot_end)

    # Iterate over slots
    curr = start_dt
    while curr < end_dt:
        slot_start = curr
        slot_end = curr + slot_delta

        # Who is working in this slot?
        active_people = set()
        for entry in schedule:
            if overlaps(entry, slot_start, slot_end):
                active_people.add(entry["person_id"])

        people_count = len(active_people)

        # Apply consumption for this slot
        for name, initial_amount in supplies.items():
            rate = rates.get(name, 0.0)
            used = people_count * rate * hours_per_slot
            total_usage[name] += used

            # Check runout
            if runout_times[name] is None and total_usage[name] > initial_amount + 1e-9:
                runout_times[name] = slot_end.isoformat()
                supplies_ok = False

        curr += slot_delta

    # Compute remaining supplies (not going below zero)
    remaining = {}
    for name, initial_amount in supplies.items():
        rem = initial_amount - total_usage[name]
        if rem < 0:
            rem = 0.0
        remaining[name] = rem

    # Compute hours_supported: earliest runout among supplies that run out
    total_horizon_hours = (end_dt - start_dt).total_seconds() / 3600.0
    if supplies_ok:
        hours_supported = total_horizon_hours
    else:
        earliest = None
        for name, t_str in runout_times.items():
            if t_str is None:
                continue
            t = datetime.fromisoformat(t_str)
            if earliest is None or t < earliest:
                earliest = t
        if earliest is None:
            hours_supported = 0.0
        else:
            hours_supported = (earliest - start_dt).total_seconds() / 3600.0

    return {
        "initial_supplies": supplies,
        "total_usage": total_usage,
        "remaining": remaining,
        "supplies_ok": supplies_ok,
        "runout_times": runout_times,
        "hours_supported": hours_supported,
    }


# --- TOOL: ScheduleBuilderTool ---
# --- TOOL: ScheduleBuilderTool ---
class ScheduleBuilderTool:
    """
    Tool that builds a staffing schedule from a disaster scenario and staffing rules.
    Input JSON shape (Action Input):
      {
        "scenario": { ... },
        "rules": { ... }
      }
    Output JSON shape (Observation):
      {
        "schedule": [ ... ]  # list of schedule entries
      }
    """

    def __init__(self):
        self.name = "schedule_builder"
        self.description = (
            "Builds a shift schedule from a disaster scenario and staffing rules, "
            "respecting max hours and simple rest constraints."
        )

    def use(self, params):
        """
        FAIRLIB passes a single argument to use(), typically a dict or JSON string.

        Expected shapes:
          params = {
            "scenario": { ... },
            "rules": { ... }
          }
        """
        # If it's coming in as a JSON string, parse it
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"schedule_builder expected JSON but got invalid string: {e}\n"
                    f"Raw input: {params!r}"
                )

        if not isinstance(params, dict):
            raise TypeError(
                f"schedule_builder expected a dict or JSON string, got {type(params)}"
            )

        if "scenario" not in params or "rules" not in params:
            raise KeyError(
                "schedule_builder input must have keys 'scenario' and 'rules'. "
                f"Got keys: {list(params.keys())}"
            )

        scenario = params["scenario"]
        rules = params["rules"]

        schedule = build_schedule_from_scenario(scenario, rules)
        return {"schedule": schedule}

# --- TOOL: SupplyMonitorTool ---
# --- TOOL: SupplyMonitorTool ---
class SupplyMonitorTool:
    """
    Tool that simulates supply usage across a schedule.
    Input JSON shape (Action Input):
      {
        "scenario": { ... },
        "schedule": [ ... ]
      }
    Output JSON shape (Observation):
      {
        "initial_supplies": {...},
        "total_usage": {...},
        "remaining": {...},
        "supplies_ok": bool,
        "runout_times": {...},
        "hours_supported": float
      }
    """

    def __init__(self):
        self.name = "supply_monitor"
        self.description = (
            "Simulates supply usage over the schedule and reports run-out times and remaining supplies."
        )

    def use(self, params):
        """
        FAIRLIB passes a single argument to use(), typically a dict or JSON string.

        Expected shape:
          params = {
            "scenario": { ... },
            "schedule": [ ... ]
          }
        """
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"supply_monitor expected JSON but got invalid string: {e}\n"
                    f"Raw input: {params!r}"
                )

        if not isinstance(params, dict):
            raise TypeError(
                f"supply_monitor expected a dict or JSON string, got {type(params)}"
            )

        if "scenario" not in params or "schedule" not in params:
            raise KeyError(
                "supply_monitor input must have keys 'scenario' and 'schedule'. "
                f"Got keys: {list(params.keys())}"
            )

        scenario = params["scenario"]
        schedule = params["schedule"]

        return simulate_supplies(scenario, schedule)



# --- AGENT BUILDER ---
def build_agent():
    llm = OpenAIAdapter(api_key=OPENAI_KEY)

    tool_registry = ToolRegistry()
    tool_registry.register_tool(CoverageCheckerTool())
    tool_registry.register_tool(ScheduleCompletenessTool())
    tool_registry.register_tool(ScheduleBuilderTool())
    tool_registry.register_tool(SupplyMonitorTool())

    print("Registered tools:", [t.name for t in tool_registry.get_all_tools().values()])

    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()
    planner = ReActPlanner(llm, tool_registry)

    planner.prompt_builder.role_definition = RoleDefinition(
    "You are a disaster response scheduling agent.\n"
    "\n"
    "Your main job is to plan and evaluate disaster relief operations.\n"
    "You have the following tools:\n"
    "- 'schedule_builder': builds a staffing schedule from a scenario (people, time horizon) and rules.\n"
    "- 'coverage_checker': checks how well a schedule covers required roles across time.\n"
    "- 'schedule_completeness': summarizes coverage results and highlights coverage gaps per time slot.\n"
    "- 'supply_monitor': simulates supply usage across the schedule and reports run-out times.\n"
    "\n"
    "MANDATORY WORKFLOW (follow in this exact order):\n"
    "1) Call 'schedule_builder' with the given scenario and rules to create a schedule.\n"
    "2) Call 'coverage_checker' with the rules, the schedule from step 1, and the horizon details.\n"
    "3) Call 'schedule_completeness' with the coverage result and horizon details to get covered_hours,\n"
    "   total_hours, and a checklist of gaps.\n"
    "4) Call 'supply_monitor' with the scenario and the schedule from step 1 to assess supplies.\n"
    "You MUST call all four tools before producing the Final Answer.\n"
    "\n"
    "INTERACTION FORMAT (CRITICAL):\n"
    "- For every intermediate reasoning step, respond in plain text using EXACTLY this pattern:\n"
    "  Thought: <short explanation of what you will do next>\n"
    "  Action: Using tool 'tool_name' with input '<JSON object as a single-line string>'\n"
    "- Do NOT wrap Thought/Action in any outer JSON object.\n"
    "- Do NOT output keys like \"thought\" or \"action\" in JSON form during intermediate steps.\n"
    "- The ONLY time you output a raw JSON object (without quotes) is in the Final Answer.\n"
    "\n"
    "EXAMPLES OF VALID INTERMEDIATE STEPS:\n"
    "  Thought: I will build the initial schedule from the scenario and rules.\n"
    "  Action: Using tool 'schedule_builder' with input '{\"scenario\": {...}, \"rules\": {...}}'\n"
    "\n"
    "  Thought: I will now check coverage using the schedule from the previous step.\n"
    "  Action: Using tool 'coverage_checker' with input '{\"rules\": {...}, \"schedule\": [...], \"horizon_start\": \"...\", \"horizon_end\": \"...\", \"slot_minutes\": 60}'\n"
    "\n"
    "FINAL ANSWER FORMAT (IMPORTANT):\n"
    "After you have used all required tools, you MUST output your Final Answer in ONE step, using:\n"
    "  Final Answer: <single JSON object with the required keys>\n"
    "\n"
    "The JSON object MUST have exactly these keys:\n"
    "{\n"
    '  \"coverage_rate\": <number between 0 and 1>,\n'
    '  \"passed\": <true or false>,\n'
    '  \"covered_hours\": <number>,\n'
    '  \"total_hours\": <number>,\n'
    '  \"first_two_gaps\": [\n'
    "    {\n"
    '      \"slot_start\": \"<ISO8601 datetime string>\",\n'
    '      \"slot_end\": \"<ISO8601 datetime string>\",\n'
    '      \"station_id\": \"<string>\",\n'
    '      \"missing_role\": \"<string>\"\n'
    "    },\n"
    "    {\n"
    '      \"slot_start\": \"<ISO8601 datetime string>\",\n'
    '      \"slot_end\": \"<ISO8601 datetime string>\",\n'
    '      \"station_id\": \"<string>\",\n'
    '      \"missing_role\": \"<string>\"\n'
    "    }\n"
    "  ],\n"
    '  \"supplies_ok\": <true or false>,\n'
    '  \"hours_supported\": <number>,\n'
    '  \"supply_runout_times\": { \"<supply_name>\": \"<ISO8601 datetime or null>\", ... },\n'
    '  \"supply_remaining\": { \"<supply_name>\": <number>, ... }\n'
    "}\n"
    "\n"
    "- If there are fewer than two gaps, 'first_two_gaps' should contain only the available gaps (0 or 1).\n"
    "- You MUST compute coverage_rate yourself as covered_hours / total_hours using the output of 'schedule_completeness'.\n"
    "- Use ONLY the output of 'supply_monitor' for supplies_ok, hours_supported, supply_runout_times, and supply_remaining.\n"
    "- Do NOT include any explanatory text outside of this JSON object in the Final Answer.\n"
)


    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        max_steps=10,
    )

    return agent


# --- HELPER: extract last JSON object from trace ---
def extract_final_json(block: str) -> str | None:
    """
    Scan the text from the end and return the last JSON object that looks like
    the FINAL ANSWER (i.e., has all required keys).
    This avoids accidentally grabbing tool inputs or other JSON snippets.
    """
    required_keys = {
        "coverage_rate",
        "passed",
        "covered_hours",
        "total_hours",
        "first_two_gaps",
        "supplies_ok",
        "hours_supported",
        "supply_runout_times",
        "supply_remaining",
    }

    def candidate_is_final(js: str) -> bool:
        try:
            obj = json.loads(js)
        except Exception:
            return False
        if not isinstance(obj, dict):
            return False
        return required_keys.issubset(obj.keys())

    end = None
    brace_depth = 0

    # Walk backward through the text, find JSON-looking chunks, and test them
    for i in range(len(block) - 1, -1, -1):
        ch = block[i]
        if ch == '}':
            if end is None:
                end = i
            brace_depth += 1
        elif ch == '{':
            brace_depth -= 1
            if brace_depth == 0 and end is not None:
                start = i
                candidate = block[start: end + 1].strip()
                if candidate_is_final(candidate):
                    return candidate
                # Reset and keep searching for an earlier JSON object
                end = None

    return None



# --- EXAMPLE DATA (rules + scenario) ---
rules = {
    "stations": [
        {
            "station_id": "command_center",
            "required": [
                {"role": "comms", "count": 1},
                {"role": "ops", "count": 1},
            ],
            "min_skill_per_role": {
                "comms": ["radio_lvl2"],
                "ops": ["ics_100"],
            },
        }
    ]
}

horizon_start = "2025-11-03T08:00:00"
horizon_end   = "2025-11-03T16:00:00"
slot_minutes  = 60


def run_disaster_sim(num_people, water_liters, food_meals, med_kits, fuel_liters):
    """
    Helper used by both CLI / GUI:
    - Builds the scenario dict from numeric inputs
    - Computes the schedule (for display)
    - Calls the agent
    - Returns (result_dict, full_trace_string, schedule_list)
    """
    agent = build_agent()

    # Simple assumption: every responder can work either comms or ops
    # at the command center and has the needed skills.
    people = []
    for i in range(num_people):
        people.append(
            {
                "person_id": f"p{i+1}",
                "roles": ["comms", "ops"],
                "skills": ["radio_lvl2", "ics_100"],
                "station_id": "command_center",
            }
        )

    scenario = {
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        "slot_minutes": slot_minutes,
        "people": people,
        "supplies": {
            "water_liters": water_liters,
            "food_meals": food_meals,
            "med_kits": med_kits,
            "fuel_liters": fuel_liters,
        },
        "location": "User-defined scenario",
    }

    # Build the schedule directly using the same logic as the tool
    # so we can display it in the GUI.
    schedule = build_schedule_from_scenario(scenario, rules)

    user_prompt = f"""
You are a disaster response scheduler.

Here are the staffing rules (Python dict):
{rules}

Here is the disaster scenario (time horizon, people, supplies):
{scenario}

Using ONLY your tools (schedule_builder, coverage_checker, schedule_completeness, supply_monitor), you MUST:
1) Build a schedule from the scenario and rules.
2) Check coverage across the horizon.
3) Evaluate schedule completeness.
4) Assess supply sufficiency for the planned schedule.

In your Final Answer (the last step), follow the system instructions and output ONLY
a single JSON object with keys:
- coverage_rate
- passed
- covered_hours
- total_hours
- first_two_gaps (array of up to two gap objects),
- supplies_ok
- hours_supported
- supply_runout_times
- supply_remaining

Do NOT add any explanation text in the Final Answer. All reasoning should occur in
your intermediate thoughts and tool calls, not in the final JSON.
"""

    full_output = asyncio.run(agent.arun(user_prompt))

    json_str = extract_final_json(full_output)
    if json_str is None:
        raise RuntimeError("Could not find a JSON Final Answer in the output.")

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse Final Answer JSON: {e}\nRaw JSON candidate was:\n{json_str}"
        ) from e

    # Now also return the schedule we computed locally
    return result, full_output, schedule


def format_hourly_schedule(schedule):
    """
    Turn the raw schedule list into human-readable hourly blocks, e.g.:

    2025-11-03 08:00 → 09:00  (responders on shift: 2)
      command_center:
        p1 (comms)
        p2 (ops)
    """
    if not schedule:
        return ["No scheduled assignments."]

    # Build a map: (start_dt, end_dt) -> { "stations": {station_id: [(person_id, role), ...]}, "people": set(...) }
    slot_map = {}

    for entry in schedule:
        start_str = entry.get("start")
        end_str = entry.get("end")
        if not start_str or not end_str:
            continue

        start_dt = datetime.fromisoformat(start_str)
        end_dt = datetime.fromisoformat(end_str)
        key = (start_dt, end_dt)

        station_id = entry.get("station_id", "unknown_station")
        person_id = entry.get("person_id", "unknown_person")
        roles = entry.get("person_roles") or []
        role = roles[0] if roles else "unknown_role"

        if key not in slot_map:
            slot_map[key] = {"stations": {}, "people": set()}

        slot_info = slot_map[key]
        slot_info["people"].add(person_id)

        station_map = slot_info["stations"]
        if station_id not in station_map:
            station_map[station_id] = []
        station_map[station_id].append((person_id, role))

    lines = []
    # Sort slots by start time
    for (start_dt, end_dt) in sorted(slot_map.keys()):
        slot_info = slot_map[(start_dt, end_dt)]
        num_people = len(slot_info["people"])
        start_fmt = start_dt.strftime("%Y-%m-%d %H:%M")
        end_fmt = end_dt.strftime("%Y-%m-%d %H:%M")

        lines.append(f"{start_fmt} → {end_fmt}  (responders on shift: {num_people})")

        # Sort stations and people for a stable, readable output
        for station_id in sorted(slot_info["stations"].keys()):
            lines.append(f"  {station_id}:")
            for person_id, role in sorted(slot_info["stations"][station_id], key=lambda t: t[0]):
                lines.append(f"    {person_id} ({role})")

        lines.append("")  # blank line between slots

    return lines


def launch_gui():
    """
    Simple tkinter GUI wrapper around run_disaster_sim().
    """
    root = tk.Tk()
    root.title("Disaster Response Scheduler")

    # Make window resizable
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    mainframe = ttk.Frame(root, padding="12 12 12 12")
    mainframe.grid(row=0, column=0, sticky="nsew")

    mainframe.columnconfigure(0, weight=0)
    mainframe.columnconfigure(1, weight=1)
    mainframe.rowconfigure(8, weight=1)  # results area expands

    # Input variables with some default values
    num_people_var = tk.StringVar(value="10")
    water_var = tk.StringVar(value="500")
    food_var = tk.StringVar(value="50")
    med_var = tk.StringVar(value="50")
    fuel_var = tk.StringVar(value="500")

    # Title
    title_label = ttk.Label(
        mainframe,
        text="Disaster Response Scenario Setup",
        font=("Segoe UI", 14, "bold"),
    )
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

    # Input fields
    ttk.Label(mainframe, text="Total responders available:").grid(
        row=1, column=0, sticky="e", padx=(0, 8), pady=2
    )
    num_entry = ttk.Entry(mainframe, textvariable=num_people_var, width=20)
    num_entry.grid(row=1, column=1, sticky="w", pady=2)

    ttk.Label(mainframe, text="Water available (liters):").grid(
        row=2, column=0, sticky="e", padx=(0, 8), pady=2
    )
    water_entry = ttk.Entry(mainframe, textvariable=water_var, width=20)
    water_entry.grid(row=2, column=1, sticky="w", pady=2)

    ttk.Label(mainframe, text="Food available (meals):").grid(
        row=3, column=0, sticky="e", padx=(0, 8), pady=2
    )
    food_entry = ttk.Entry(mainframe, textvariable=food_var, width=20)
    food_entry.grid(row=3, column=1, sticky="w", pady=2)

    ttk.Label(mainframe, text="Medical kits available:").grid(
        row=4, column=0, sticky="e", padx=(0, 8), pady=2
    )
    med_entry = ttk.Entry(mainframe, textvariable=med_var, width=20)
    med_entry.grid(row=4, column=1, sticky="w", pady=2)

    ttk.Label(mainframe, text="Fuel available (liters):").grid(
        row=5, column=0, sticky="e", padx=(0, 8), pady=2
    )
    fuel_entry = ttk.Entry(mainframe, textvariable=fuel_var, width=20)
    fuel_entry.grid(row=5, column=1, sticky="w", pady=2)

    # Run button
    run_button = ttk.Button(mainframe, text="Run Scheduler")
    run_button.grid(row=6, column=0, columnspan=2, pady=(8, 8))

    # Results area
    results_label = ttk.Label(mainframe, text="Results:")
    results_label.grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 2))

    results_text = ScrolledText(mainframe, width=80, height=20, wrap="word")
    results_text.grid(row=8, column=0, columnspan=2, sticky="nsew")
    results_text.configure(state="disabled")

    def on_run():
        # Parse inputs
        try:
            num_people = int(num_people_var.get())
            water = float(water_var.get())
            food = float(food_var.get())
            med = float(med_var.get())
            fuel = float(fuel_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numeric values for all fields.")
            return

        if num_people <= 0:
            messagebox.showerror("Input error", "Total responders must be greater than zero.")
            return

        # Call the core agent logic
        try:
            result, _trace, schedule = run_disaster_sim(num_people, water, food, med, fuel)
        except Exception as e:
            messagebox.showerror("Scheduler error", str(e))
            return

        lines = []

        # === Supply Summary at the top ===
        lines.append("=== Supply Summary ===")
        lines.append(f"Supplies OK for full horizon? {result.get('supplies_ok')}")
        lines.append(f"Hours supported by supplies:  {result.get('hours_supported')}")
        lines.append("")
        lines.append("Remaining supplies:")
        for name, rem in (result.get("supply_remaining") or {}).items():
            lines.append(f"  {name}: {rem}")
        lines.append("")
        lines.append("Runout times:")
        for name, t_str in (result.get("supply_runout_times") or {}).items():
            status = t_str if t_str is not None else "did not run out"
            lines.append(f"  {name}: {status}")

        # === Schedule Summary next ===
        lines.append("")
        lines.append("=== Schedule Summary ===")
        lines.append(f"Passed?        {result.get('passed')}")
        cov_rate = result.get("coverage_rate")
        cov_hours = result.get("covered_hours")
        total_hours = result.get("total_hours")
        lines.append(f"Coverage rate: {cov_rate} ({cov_hours}/{total_hours} hours)")

        gaps = result.get("first_two_gaps") or []
        if gaps:
            lines.append("")
            lines.append("First gaps:")
            for i, gap in enumerate(gaps, start=1):
                lines.append(
                    f"  Gap {i}: {gap.get('slot_start')} to {gap.get('slot_end')} "
                    f"at {gap.get('station_id')} missing role '{gap.get('missing_role')}'"
                )
        else:
            lines.append("")
            lines.append("No uncovered gaps in first_two_gaps.")

        # === Hourly Schedule with more detail ===
        lines.append("")
        lines.append("=== Hourly Schedule ===")
        lines.extend(format_hourly_schedule(schedule))

        # Update the text box
        results_text.configure(state="normal")
        results_text.delete("1.0", tk.END)
        results_text.insert(tk.END, "\n".join(lines))
        results_text.configure(state="disabled")


    run_button.configure(command=on_run)
    num_entry.focus()

    root.mainloop()



# --- MAIN ---
if __name__ == "__main__":
    launch_gui()
