# project_tools/coverage_checker_tool.py
import json
from datetime import datetime, timedelta
from collections import defaultdict

class CoverageCheckerTool:
    """
    name: coverage_checker
    description: |
      Evaluate schedule coverage per time slot. Input MUST be JSON with:
      {
        "rules": { "stations": [ { "station_id": str, "required": [{"role": str, "count": int}], "min_skill_per_role": {role: [skills]} } ] },
        "schedule": [ { "person_id": str, "person_roles": [str], "person_skills": [str], "station_id": str, "start": ISO8601, "end": ISO8601 } ],
        "horizon_start": ISO8601,
        "horizon_end": ISO8601,
        "slot_minutes": int (optional, default 60)
      }
      Returns JSON with {"coverage_rate": float, "gaps": [...] } where gaps list missing roles per slot.
    """
    name = "coverage_checker"
    description = __doc__

    def use(self, tool_input: str) -> str:
        try:
            data = json.loads(tool_input)
            rules = data["rules"]; schedule = data["schedule"]
            horizon_start = data["horizon_start"]; horizon_end = data["horizon_end"]
            slot_minutes = int(data.get("slot_minutes", 60))

            def parse_iso(ts): return datetime.fromisoformat(ts)
            def overlaps(a1, a2, b1, b2): return a1 < b2 and b1 < a2

            # Generate time slots
            t = parse_iso(horizon_start); end = parse_iso(horizon_end)
            delta = timedelta(minutes=slot_minutes)
            slots = []
            while t < end:
                slots.append((t, t + delta))
                t += delta

            # Map availability per station/slot/role
            avail = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for a in schedule:
                p = a["person_id"]; roles = a.get("person_roles", [])
                skills = set(a.get("person_skills", []))
                st = a["station_id"]
                a1, a2 = parse_iso(a["start"]), parse_iso(a["end"])
                for i, (s, e) in enumerate(slots):
                    if overlaps(a1, a2, s, e):
                        for r in roles:
                            avail[st][i][r].append({"person": p, "skills": skills})

            # Evaluate coverage
            gaps, total, filled = [], 0, 0
            for st in rules["stations"]:
                sid = st["station_id"]
                for i, (s, e) in enumerate(slots):
                    for req in st["required"]:
                        role = req["role"]; need = req["count"]; total += need
                        got = avail[sid][i].get(role, [])
                        skill_need = set(st.get("min_skill_per_role", {}).get(role, []))
                        qual = [g for g in got if skill_need.issubset(g["skills"])]
                        have = len(qual); filled += min(need, have)
                        if have < need:
                            gaps.append({
                                "station_id": sid,
                                "slot_start": s.isoformat(),
                                "slot_end": e.isoformat(),
                                "missing": {"role": role, "need": need, "have": have}
                            })
            rate = round(filled / total, 3) if total else 0.0
            return json.dumps({"coverage_rate": rate, "gaps": gaps})
        except Exception as e:
            return f"Error: {e}"
