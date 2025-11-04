import os, json, asyncio
from dotenv import load_dotenv
from fairlib import SimpleAgent, ToolRegistry, ToolExecutor, ReActPlanner, OpenAIAdapter, WorkingMemory
from project_tools.coverage_checker_tool import CoverageCheckerTool
from project_tools.schedule_completeness_tool import ScheduleCompletenessTool

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY missing — set it in .env or $env:OPENAI_API_KEY")

def build_agent():
    llm = OpenAIAdapter(api_key=OPENAI_KEY)
    registry = ToolRegistry()
    registry.register_tool(CoverageCheckerTool())
    registry.register_tool(ScheduleCompletenessTool())
    executor = ToolExecutor(registry)
    memory = WorkingMemory()
    planner = ReActPlanner(llm, registry)
    planner.prompt_builder.role_definition.text = (
        "You are a disaster response scheduler. Always use tools 'coverage_checker' "
        "then 'schedule_completeness' with the JSON I provide before answering."
    )
    return SimpleAgent(llm=llm, planner=planner, tool_executor=executor, memory=memory)

if __name__ == "__main__":
    agent = build_agent()

    rules = {"stations":[{"station_id":"command_center",
                           "required":[{"role":"comms","count":1},{"role":"ops","count":1}],
                           "min_skill_per_role":{"comms":["radio_lvl2"],"ops":["ics_100"]}}]}
    schedule = [
      {"person_id":"p1","person_roles":["comms"],"person_skills":["radio_lvl2"],
       "station_id":"command_center","start":"2025-11-03T08:00:00","end":"2025-11-03T12:00:00"},
      {"person_id":"p2","person_roles":["ops"],"person_skills":["ics_100"],
       "station_id":"command_center","start":"2025-11-03T12:00:00","end":"2025-11-03T16:00:00"}
    ]
    horizon_start = "2025-11-03T08:00:00"
    horizon_end   = "2025-11-03T16:00:00"


    prompt = f"""
First call tool 'coverage_checker' with this JSON:
{{"rules": {json.dumps(rules)}, "schedule": {json.dumps(schedule)},
  "horizon_start":"{horizon_start}", "horizon_end":"{horizon_end}", "slot_minutes":60}}
Then call 'schedule_completeness' with:
{{"coverage_result": <use previous tool JSON>, "horizon_start":"{horizon_start}",
  "horizon_end":"{horizon_end}", "slot_minutes":60}}
Return: coverage_rate, passed true/false, covered_hours/total_hours, first 2 gaps.
"""
    print(asyncio.run(agent.arun(prompt)))
