
from .agent import Agent
from .code_agent import CoderAgent
from .casual_agent import CasualAgent
from .file_agent import FileAgent
from .general_agent import GeneralAgent
from .planner_agent import PlannerAgent
from .mcp_agent import McpAgent

try:
    from .browser_agent import BrowserAgent
except ImportError:
    BrowserAgent = None  # selenium not installed (slim deps)

__all__ = ["Agent", "CoderAgent", "CasualAgent", "FileAgent", "PlannerAgent", "BrowserAgent", "McpAgent", "GeneralAgent"]
