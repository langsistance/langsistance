import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from sources.dynamic_tool_params import execute_backend_tool_request
from sources.knowledge.knowledge import KnowledgeItem, ToolItem, get_knowledge_by_id, get_tool_by_id
from sources.logger import Logger


logger = Logger("workflow_executor.log")


@dataclass
class WorkflowStepResult:
    step_id: str
    knowledge: KnowledgeItem
    tool: ToolItem
    params: Dict[str, Any]
    data: Any


@dataclass
class WorkflowResult:
    final_data: Any
    raw_items: list | None
    steps: List[WorkflowStepResult]
    workflow_question: str = ""
    workflow_instructions: str = ""
    output_mode: str = "last"  # "last" | "all"


def is_workflow_knowledge(knowledge_item: KnowledgeItem | None) -> bool:
    return bool(knowledge_item and getattr(knowledge_item, "type", 1) == 2)


def parse_workflow_spec(workflow_spec: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(workflow_spec, dict):
        spec = workflow_spec
    else:
        spec = json.loads(workflow_spec or "{}")

    if spec.get("type") != "workflow":
        raise ValueError("workflow spec type must be workflow")
    if spec.get("mode", "context_chain") != "context_chain":
        raise ValueError("only context_chain workflow mode is supported")

    steps = spec.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("workflow spec must include at least one step")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"workflow step {index + 1} must be an object")
        if not step.get("knowledge_id"):
            raise ValueError(f"workflow step {index + 1} must include knowledge_id")
    return spec


class WorkflowExecutor:
    def __init__(
        self,
        llm: Any,
        knowledge_resolver: Callable[[int], KnowledgeItem | None] = get_knowledge_by_id,
        tool_resolver: Callable[[int], ToolItem | None] = get_tool_by_id,
        tool_executor: Callable[[ToolItem, Dict[str, Any]], Dict[str, Any]] = execute_backend_tool_request,
    ):
        self.llm = llm
        self.knowledge_resolver = knowledge_resolver
        self.tool_resolver = tool_resolver
        self.tool_executor = tool_executor

    async def execute(
        self,
        workflow_spec: str | Dict[str, Any],
        user_prompt: str,
        workflow_knowledge: KnowledgeItem | None = None,
    ) -> WorkflowResult:
        spec = parse_workflow_spec(workflow_spec)
        output_mode = spec.get("output_mode", "last")
        if output_mode not in ("last", "all"):
            output_mode = "last"
        previous_results: List[Dict[str, Any]] = []
        step_results: List[WorkflowStepResult] = []
        raw_items = None
        final_data = None
        workflow_question = getattr(workflow_knowledge, "question", "") if workflow_knowledge else ""
        workflow_instructions = getattr(workflow_knowledge, "answer", "") if workflow_knowledge else ""

        for index, step in enumerate(spec["steps"], start=1):
            knowledge = self.knowledge_resolver(int(step["knowledge_id"]))
            if not knowledge:
                raise ValueError(f"workflow step {index} knowledge not found")
            if not knowledge.tool_id:
                raise ValueError(f"workflow step {index} knowledge has no linked tool")
            logger.info(f"workflow step {index} knowledge: {knowledge}")

            tool = self.tool_resolver(int(knowledge.tool_id))
            if not tool:
                raise ValueError(f"workflow step {index} tool not found")
            if tool.push != 2:
                raise ValueError("workflow context_chain only supports backend tools with push=2")
            logger.info(f"workflow step {index} tool: {tool}")

            params = await self._generate_tool_params(
                user_prompt=user_prompt,
                step_index=index,
                total_steps=len(spec["steps"]),
                knowledge=knowledge,
                tool=tool,
                previous_results=previous_results,
                workflow_question=workflow_question,
                workflow_instructions=workflow_instructions,
            )
            logger.info(f"workflow step {index} params: {params}")

            # LLM signals that previous step(s) did not provide the data this
            # step needs — terminate early with the human-readable message.
            if params.get("_terminate"):
                logger.info(
                    f"workflow step {index} LLM requested early termination: "
                    f"{params.get('message', '')}"
                )
                final_data = params.get("message", "未查询到相关数据，无法继续后续步骤。")
                raw_items = None  # no tool was executed at this step
                step_result = WorkflowStepResult(
                    step_id=str(step.get("id") or f"step_{index}"),
                    knowledge=knowledge,
                    tool=tool,
                    params={},
                    data=final_data,
                )
                step_results.append(step_result)
                break

            tool_result = self.tool_executor(tool, params)
            logger.info(f"workflow step {index}")
            final_data = tool_result.get("data")
            raw_items = tool_result.get("raw_items")

            step_result = WorkflowStepResult(
                step_id=str(step.get("id") or f"step_{index}"),
                knowledge=knowledge,
                tool=tool,
                params=params,
                data=final_data,
            )
            step_results.append(step_result)
            previous_results.append({
                "step_id": step_result.step_id,
                "knowledge_question": knowledge.question,
                "tool_title": tool.title,
                "result": final_data,
            })

        return WorkflowResult(
            final_data=final_data,
            raw_items=raw_items,
            steps=step_results,
            workflow_question=workflow_question,
            workflow_instructions=workflow_instructions,
            output_mode=output_mode,
        )

    async def _generate_tool_params(
        self,
        user_prompt: str,
        step_index: int,
        total_steps: int,
        knowledge: KnowledgeItem,
        tool: ToolItem,
        previous_results: List[Dict[str, Any]],
        workflow_question: str = "",
        workflow_instructions: str = "",
    ) -> Dict[str, Any]:
        # Step 1 always executes: derive params from the user request and
        # knowledge/tool template. Only subsequent steps may terminate early
        # when prior results don't contain the data they need.
        if step_index == 1:
            termination_rule = ""
        else:
            termination_rule = (
                "CRITICAL: If previous step results are empty or do NOT contain the data required "
                "to fill the current tool parameters, do NOT invent missing values — instead return "
                '{"_terminate": true, "message": "a user-friendly Chinese sentence explaining that '
                'no matching data was found and the workflow cannot continue"}. '
            )
        system_prompt = (
            "You generate JSON parameters for one backend API tool in a composed knowledge workflow. "
            "Return only a JSON object. The object may contain path, query, and body. "
            "Use the user request and the current knowledge instructions to fill parameters. "
            + (
                "Also use previous step results when available. "
                "If a needed value is present in previous results, extract it exactly. "
                if step_index > 1 else
                ""
            ) +
            "If the original tool params contain api-key, api_key, apikey, x-api-key, "
            "or another API key field, preserve that key and its value exactly in the generated params. "
            "Do not invent values. "
            + termination_rule
        )
        payload = {
            "workflow": {
                "question": workflow_question,
                "instructions": workflow_instructions,
            },
            "step": {
                "index": step_index,
                "total": total_steps,
            },
            "user_request": user_prompt,
            "current_knowledge": {
                "question": knowledge.question,
                "answer": knowledge.answer,
            },
            "tool": {
                "title": tool.title,
                "description": tool.description,
                "params_template": tool.params,
            },
        }
        if previous_results:
            payload["previous_results"] = previous_results
        user_content = json.dumps(payload, ensure_ascii=False, indent=2)
        params = await self.llm.complete_json(system_prompt, user_content)
        if not isinstance(params, dict):
            raise ValueError("workflow generated tool params must be a JSON object")
        logger.info(f"workflow step {step_index} generated params: {params}")
        return params
