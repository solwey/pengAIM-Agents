"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import inspect
import json
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

import graphs.open_deep_research.utils as utils
from graphs.open_deep_research.configuration import (
    AgentMode,
    Configuration,
)
from graphs.open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt_online,
    compress_research_system_prompt_rag,
    final_report_generation_prompt_online,
    final_report_generation_prompt_rag,
    lead_researcher_prompt_online,
    lead_researcher_prompt_rag,
    research_system_prompt_online,
    research_system_prompt_rag,
    transform_messages_into_research_topic_prompt,
)
from graphs.open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


async def provide_placeholders(state: AgentState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    steps = configurable.steps or []

    last_message = state.get("messages", [])[-1]

    if not steps:
        return Command(goto="prepare_step")

    # Parse the message to check for "Start" format with inline placeholders
    message_text = ""
    if isinstance(last_message, HumanMessage):
        message_text = (
            last_message.text()
            if hasattr(last_message, "text")
            else str(last_message.content)
        )

    is_start_message, all_steps_placeholders = utils.parse_start_message(message_text)

    if isinstance(last_message, HumanMessage) and not is_start_message:
        return Command(goto="clarify_with_user", update={"step_index": -1})

    step_index = state.get("step_index", 0)
    if step_index >= len(steps):
        return Command(goto=END)

    current_step = steps[step_index]

    has_sub_prompts = bool(
        current_step.parallel_sub_prompts or current_step.sequential_sub_prompts
    )
    goto = "run_sub_prompts" if has_sub_prompts else "prepare_step"

    expected_placeholders = [
        str(x).strip() for x in current_step.placeholders if str(x).strip()
    ]
    expected_placeholders = list(dict.fromkeys(expected_placeholders))

    if not expected_placeholders:
        return Command(goto=goto)

    # Check if we have inline placeholders from the "Start" message
    # Format: Start [{"company_name": "Acme", "context": "..."}, {"other_field": "value"}]
    # Each array element corresponds to a step by index
    if all_steps_placeholders and step_index < len(all_steps_placeholders):
        step_placeholders_dict = all_steps_placeholders[step_index]

        if isinstance(step_placeholders_dict, dict):
            # Check if all expected placeholders are provided for this step
            all_provided = all(
                field in step_placeholders_dict for field in expected_placeholders
            )

            if all_provided:
                # Convert dict format to list of {"field": key, "value": val} for normalize_placeholders
                placeholders_list = [
                    {"field": key, "value": value}
                    for key, value in step_placeholders_dict.items()
                ]
                # Use the inline placeholders directly, no interrupt needed
                return Command(
                    goto=goto,
                    update={
                        "placeholders": {
                            "type": "override",
                            "value": utils.normalize_placeholders(placeholders_list),
                        },
                    },
                )

    # Fall back to interrupt if placeholders not provided inline
    human_payload = interrupt(
        {
            "type": "placeholders_required",
            "expected": expected_placeholders,
            "step_index": step_index,
        }
    )

    if isinstance(human_payload, dict) and human_payload.get("skip"):
        return Command(
            goto="provide_placeholders",
            update={
                "messages": [
                    ToolMessage(
                        content=f"Step {step_index + 1} skipped by user",
                        name="Placeholders",
                        tool_call_id=f"placeholders:step:{step_index}:skip",
                    )
                ],
                "placeholders": {"type": "override", "value": []},
                "step_index": step_index + 1,
            },
        )

    return Command(
        goto=goto,
        update={
            "placeholders": {
                "type": "override",
                "value": utils.normalize_placeholders(human_payload),
            },
        },
    )


async def run_sub_prompts(state: AgentState, config: RunnableConfig):
    """Unified handler for sequential + parallel sub-prompts with a two-phase flow."""
    configurable = Configuration.from_runnable_config(config)
    steps = configurable.steps or []

    step_index = state.get("step_index", 0)
    if step_index >= len(steps):
        return Command(goto="prepare_step")

    step = steps[step_index]
    if not (step.sequential_sub_prompts or step.parallel_sub_prompts):
        return Command(goto="prepare_step")

    phase = state.get("sub_prompts_phase", "announce")
    user_placeholders = state.get("placeholders", []) or []

    if phase == "announce":
        # Build deterministic metadata for both kinds of sub-prompts
        used: set[str] = set()
        seq_meta: list[dict] = []
        par_meta: list[dict] = []
        ui_messages: list[ToolMessage] = []

        for it in step.sequential_sub_prompts or []:
            name = utils.normalize_branch_name(it.name, used)
            preview = utils.apply_placeholders(it.text, user_placeholders)
            ui_messages.append(
                ToolMessage(
                    content=f"Starting sequential subprompt: {name}\nTopic: {utils.truncate_result(preview)}",
                    name="SequentialSubprompt",
                    tool_call_id=f"sequential:{name}:start",
                )
            )
            seq_meta.append({"name": name, "text_template": it.text})

        for it in step.parallel_sub_prompts or []:
            name = utils.normalize_branch_name(it.name, used)
            preview = utils.apply_placeholders(it.text, user_placeholders)
            ui_messages.append(
                ToolMessage(
                    content=f"Starting parallel subprompt: {name}\nTopic: {utils.truncate_result(preview)}",
                    name="ParallelSubprompt",
                    tool_call_id=f"parallel:{name}:start",
                )
            )
            par_meta.append({"name": name, "text_template": it.text})

        return Command(
            goto="run_sub_prompts",
            update={
                "messages": ui_messages,
                "sub_prompts_phase": "execute",
                "sequential_branches": seq_meta,
                "parallel_branches": par_meta,
                "sequential_context": [],
                "synthetic_placeholders": [],
            },
        )

    # Phase == execute
    seq_meta = state.get("sequential_branches", []) or []
    par_meta = state.get("parallel_branches", []) or []

    # 1) Execute sequential
    context_chunks = state.get("sequential_context", []) or []
    synthetic_placeholders = state.get("synthetic_placeholders", []) or []
    ui_messages: list[ToolMessage] = []

    for item in seq_meta:
        name = item.get("name", "subprompt")
        text = item.get("text_template", "")
        merged_ph = list(user_placeholders) + synthetic_placeholders
        base_text = utils.apply_placeholders(text, merged_ph)
        ctx_text = (
            ("\n\nContext from previous steps:\n" + "\n\n".join(context_chunks))
            if context_chunks
            else ""
        )
        final_text = base_text + ctx_text

        try:
            observation = await researcher_subgraph.ainvoke(
                {
                    "researcher_messages": [HumanMessage(content=final_text)],
                    "research_topic": final_text,
                },
                config,
            )
            compressed = observation.get("compressed_research", "")
        except Exception as e:
            compressed = f"Error during sequential subprompt '{name}': {e}"
        context_chunks.append(compressed)
        ui_messages.append(
            ToolMessage(
                content=f"Completed: {name}\nShort result:\n{utils.truncate_result(compressed)}",
                name="SequentialSubprompt",
                tool_call_id=f"sequential:{name}:done",
            )
        )
        synthetic_placeholders.append({"field": name, "value": compressed})

    # 2) Execute parallel
    parallel_sends: list[Send] = []

    merged_for_parallel = user_placeholders + synthetic_placeholders
    for item in par_meta:
        name = item.get("name", "")
        text = item.get("text_template", "")
        if not name:
            continue
        parallel_sends.append(
            Send(
                "parallel_subprompt",
                {
                    "subprompt_name": name,
                    "subprompt_template": text,
                    "placeholders": {"type": "override", "value": merged_for_parallel},
                },
            )
        )

    ui_messages.append(
        ToolMessage(
            content=f"Parallel phase started: {len(par_meta)} subprompts scheduled",
            name="ParallelSubprompt",
            tool_call_id="parallel:__batch:start",
        )
    )

    # Hand off to a small fan-out node that will emit these Sends
    return Command(
        goto="fan_out_parallel",
        update={
            "messages": ui_messages,
            "subprompt_results": {},
            "parallel_sends": parallel_sends,
            "pending_parallel": len(par_meta),
            "synthetic_placeholders": synthetic_placeholders,
            "sequential_context": context_chunks,
        },
    )


async def fan_out_parallel(state: AgentState, config: RunnableConfig):
    """Fan-out trigger node. This node does not modify state."""
    return {}


async def parallel_subprompt(state: AgentState, config: RunnableConfig):
    """Worker for a single parallel subprompt (one subprompt per node call)."""
    nm = state.get("subprompt_name", "subprompt")
    tmpl = state.get("subprompt_template", "")
    placeholders = state.get("placeholders", []) or []

    sub_text = utils.apply_placeholders(tmpl, placeholders)

    try:
        observation = await researcher_subgraph.ainvoke(
            {
                "researcher_messages": [HumanMessage(content=sub_text)],
                "research_topic": sub_text,
            },
            config,
        )
        comp = observation.get("compressed_research", "")
    except Exception as e:
        comp = f"Error during parallel subprompt '{nm}': {e}"

    ui_msg = ToolMessage(
        content=f"Completed: {nm}\nShort result:\n{utils.truncate_result(comp)}",
        name="ParallelSubprompt",
        tool_call_id=f"parallel:{nm}:done",
    )

    return {
        "messages": [ui_msg],
        "last_parallel_result": [{"name": nm, "compressed": comp}],
    }


async def collect_parallel(state: AgentState, config: RunnableConfig):
    """Accumulate results from parallel_subprompt workers; continue when all done."""
    result_map: dict = state.get("subprompt_results", {}) or {}
    pending: int = state.get("pending_parallel", 0) or 0

    batch: list = state.get("last_parallel_result", []) or []
    for item in batch:
        nm = item.get("name")
        comp = item.get("compressed")
        if nm is not None:
            result_map[nm] = comp
            pending = max(0, pending - 1)

    if pending > 0:
        return {
            "subprompt_results": result_map,
            "pending_parallel": pending,
            "last_parallel_result": [],
        }

    synthetic_placeholders: list = state.get("synthetic_placeholders", []) or []
    user_placeholders: list = state.get("placeholders", []) or []
    existing = {
        p["field"] for p in user_placeholders if isinstance(p, dict) and "field" in p
    }
    for k, v in result_map.items():
        if k not in existing:
            synthetic_placeholders.append({"field": k, "value": v})

    merged_placeholders = user_placeholders + [
        sp for sp in synthetic_placeholders if sp.get("field") not in existing
    ]

    return Command(
        goto="prepare_step",
        update={
            "subprompt_results": result_map,
            "placeholders": {"type": "override", "value": merged_placeholders},
            "sequential_context": state.get("sequential_context", []) or [],
            "parallel_sends": None,
            "pending_parallel": 0,
            "last_parallel_result": [],
            "synthetic_placeholders": [],
        },
    )


async def prepare_step(state: AgentState, config: RunnableConfig):
    """Prepares a step from the step's configuration, if any."""
    configurable = Configuration.from_runnable_config(config)
    steps = configurable.steps or []

    if not steps:
        return Command(goto="clarify_with_user")

    step_index = state.get("step_index", 0)
    if step_index >= len(steps):
        return Command(goto=END)

    step = steps[step_index]
    prompt_text = utils.apply_placeholders(step.text, state.get("placeholders", []))

    return Command(
        goto="clarify_with_user",
        update={
            "messages": [HumanMessage(content=prompt_text)],
            "step_index": step_index,
            "placeholders": {"type": "override", "value": []},
        },
    )


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.

    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences

    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")

    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    api_key = await utils.get_api_key_for_model(configurable.research_model, config)
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": api_key,
        "tags": ["langsmith:nostream"],
    }

    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model.with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # Step 3: Analyze whether clarification is needed
    sanitized_messages = utils.apply_security_to_model_messages(messages, config)
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(sanitized_messages),
        date=utils.get_today_str(),
    )
    secured = utils.apply_security_to_model_messages(
        [HumanMessage(content=prompt_content)],
        config,
    )
    response = await clarification_model.ainvoke(secured)

    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]},
        )


async def write_research_brief(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.

    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings

    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    api_key = await utils.get_api_key_for_model(configurable.research_model, config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": api_key,
        "tags": ["langsmith:nostream"],
    }

    # Configure model for structured research question generation
    research_model = (
        configurable_model.with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate structured research brief from user messages
    history_messages = state.get("messages", [])
    sanitized_history_messages = utils.apply_security_to_model_messages(
        history_messages, config
    )
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(sanitized_history_messages),
        date=utils.get_today_str(),
    )
    secured = utils.apply_security_to_model_messages(
        [HumanMessage(content=prompt_content)],
        config,
    )
    response = await research_model.ainvoke(secured)

    # Step 3: Initialize supervisor with research brief and instructions
    agent_mode = utils.get_agent_mode(config)
    supervisor_prompt_template = (
        lead_researcher_prompt_rag
        if agent_mode == AgentMode.RAG
        else lead_researcher_prompt_online
    )

    supervisor_system_prompt = supervisor_prompt_template.format(
        date=utils.get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
    )
    messages = []
    if configurable.system_prompt:
        messages.append(SystemMessage(content=configurable.system_prompt))
    messages.append(SystemMessage(content=supervisor_system_prompt))
    if configurable.sales_context_prompt:
        messages.append(SystemMessage(content=configurable.sales_context_prompt))
    messages.append(HumanMessage(content=response.research_brief))

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": {"type": "override", "value": response.research_brief},
            "supervisor_messages": {"type": "override", "value": messages},
        },
    )


async def supervisor(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.

    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.

    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings

    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    api_key = await utils.get_api_key_for_model(configurable.research_model, config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": api_key,
        "tags": ["langsmith:nostream"],
    }

    agent_mode = utils.get_agent_mode(config)

    # Choose base tools by mode: ConductResearch only in ONLINE
    if agent_mode == AgentMode.RAG:
        base_tools = [ResearchComplete, utils.think_tool]
    else:
        base_tools = [ConductResearch, ResearchComplete, utils.think_tool]

    dynamic_tools = await utils.get_all_tools(config)
    lead_researcher_tools = base_tools + dynamic_tools

    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model.bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    secured = utils.apply_security_to_model_messages(supervisor_messages, config)
    response = await research_model.ainvoke(secured)
    if isinstance(response, AIMessage):
        response = utils.apply_security_to_model_output(response, config)

    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.

    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase

    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings

    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    rag_mode = utils.get_agent_mode(config) == AgentMode.RAG

    # Define exit criteria for research phase
    exceeded_allowed_iterations = (
        research_iterations > configurable.max_researcher_iterations
    )
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": utils.get_notes_from_tool_calls(supervisor_messages),
                "research_brief": {
                    "type": "override",
                    "value": state.get("research_brief", ""),
                },
            },
        )

    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(
            ToolMessage(
                content=f"Reflection recorded: {reflection_content}",
                name="think_tool",
                tool_call_id=tool_call["id"],
            )
        )

    other_tool_calls = [
        tc
        for tc in most_recent_message.tool_calls
        if tc["name"] not in {"think_tool", "ConductResearch", "ResearchComplete"}
    ]

    if other_tool_calls:
        # Execute via the same safe executor used by researchers
        tools = await utils.get_all_tools(config)
        tools_by_name = {
            (getattr(t, "name", None) or t.get("name")): t
            for t in tools
            if (getattr(t, "name", None) or t.get("name"))
        }

        exec_tasks = []
        for tc in other_tool_calls:
            tool_obj = tools_by_name.get(tc["name"])
            if tool_obj is None:
                all_tool_messages.append(
                    ToolMessage(
                        content=f"Error: Tool '{tc['name']}' is not available to the supervisor.",
                        name=tc["name"],
                        tool_call_id=tc["id"],
                    )
                )
                continue
            exec_tasks.append(execute_tool_safely(tool_obj, tc["args"], config))

        if exec_tasks:
            results = await asyncio.gather(*exec_tasks)
            for result, tc in zip(results, other_tool_calls, strict=False):
                name = tc["name"]
                args = tc.get("args") or {}

                tool_content = result

                if name == "rag_search":
                    tool_content = json.dumps(
                        {"answer": result, "question": args.get("query")},
                        ensure_ascii=False,
                    )

                all_tool_messages.append(
                    ToolMessage(
                        content=utils.apply_security_to_tool_result(
                            tool_content, name, tc["id"], config
                        ),
                        name=name,
                        tool_call_id=tc["id"],
                        additional_kwargs={"visibility": "internal"},
                    )
                )

    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        if rag_mode:
            # In RAG mode, ConductResearch is not permitted
            for tool_call in conduct_research_calls:
                all_tool_messages.append(
                    ToolMessage(
                        content=(
                            "Error: ConductResearch is not available in RAG mode. "
                            "Disable RAG to delegate web-based research tasks."
                        ),
                        name="ConductResearch",
                        tool_call_id=tool_call["id"],
                    )
                )
        else:
            try:
                # Limit concurrent research units to prevent resource exhaustion
                allowed_conduct_research_calls = conduct_research_calls[
                    : configurable.max_concurrent_research_units
                ]
                overflow_conduct_research_calls = conduct_research_calls[
                    configurable.max_concurrent_research_units :
                ]

                # Execute research tasks in parallel
                research_tasks = [
                    researcher_subgraph.ainvoke(
                        {
                            "researcher_messages": [
                                HumanMessage(
                                    content=tool_call["args"]["research_topic"]
                                )
                            ],
                            "research_topic": tool_call["args"]["research_topic"],
                        },
                        config,
                    )
                    for tool_call in allowed_conduct_research_calls
                ]

                tool_results = await asyncio.gather(*research_tasks)

                # Create tool messages with research results
                for observation, tool_call in zip(
                    tool_results, allowed_conduct_research_calls, strict=False
                ):
                    all_tool_messages.append(
                        ToolMessage(
                            content=utils.apply_security_to_tool_result(
                                observation.get(
                                    "compressed_research",
                                    "Error synthesizing research report: Maximum retries exceeded",
                                ),
                                tool_call["name"],
                                tool_call["id"],
                                config,
                            ),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )

                # Handle overflow research calls with error messages
                for overflow_call in overflow_conduct_research_calls:
                    all_tool_messages.append(
                        ToolMessage(
                            content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                            name="ConductResearch",
                            tool_call_id=overflow_call["id"],
                        )
                    )

                # Aggregate raw notes from all research results
                raw_notes_concat = "\n".join(
                    [
                        "\n".join(observation.get("raw_notes", []))
                        for observation in tool_results
                    ]
                )

                if raw_notes_concat:
                    update_payload["raw_notes"] = [raw_notes_concat]

            except Exception as e:
                # Handle research execution errors
                if utils.is_token_limit_exceeded(e, configurable.research_model):
                    # Token limit exceeded - end research phase
                    return Command(
                        goto=END,
                        update={
                            "notes": utils.get_notes_from_tool_calls(
                                supervisor_messages
                            ),
                            "research_brief": {
                                "type": "override",
                                "value": state.get("research_brief", ""),
                            },
                        },
                    )
                else:
                    # For other errors, continue the loop with an error message
                    all_tool_messages.append(
                        ToolMessage(
                            content=f"Error executing ConductResearch: {e}",
                            name="ConductResearch",
                            tool_call_id=conduct_research_calls[0]["id"],
                        )
                    )

    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(goto="supervisor", update=update_payload)


# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)  # Main supervisor logic
supervisor_builder.add_node(
    "supervisor_tools", supervisor_tools
)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()


async def researcher(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.

    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.

    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability

    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    # Get all available research tools (search, think_tool)
    tools = await utils.get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure your search API"
        )

    # Step 2: Configure the researcher model with tools
    api_key = await utils.get_api_key_for_model(configurable.research_model, config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": api_key,
        "tags": ["langsmith:nostream"],
    }

    # Choose mode-specific researcher prompt
    agent_mode = utils.get_agent_mode(config)
    prompt_template = (
        research_system_prompt_rag
        if agent_mode == AgentMode.RAG
        else research_system_prompt_online
    )
    researcher_prompt = prompt_template.format(
        date=utils.get_today_str(),
        search_tool=configurable.search_api.value,
    )

    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model.bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)]
    if configurable.sales_context_prompt:
        messages.append(SystemMessage(content=configurable.sales_context_prompt))
    messages += researcher_messages

    secured = utils.apply_security_to_model_messages(messages, config)
    response = await research_model.ainvoke(secured)
    if isinstance(response, AIMessage):
        response = utils.apply_security_to_model_output(response, config)

    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        },
    )


# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        call = getattr(tool, "ainvoke", None) or getattr(tool, "invoke", None)
        if call is None:
            raise RuntimeError(f"Tool {tool} has neither ainvoke nor invoke")

        result = call(args, config)

        if inspect.isawaitable(result):
            return await result
        return result
    except Exception as e:
        print(e)
        return f"Error executing tool: {str(e)}"


async def researcher_tools(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.

    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, firecrawl_search, web_search) - Information gathering
    3. ResearchComplete - Signals completion of individual research task

    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings

    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = utils.openai_websearch_called(
        most_recent_message
    ) or utils.anthropic_websearch_called(most_recent_message)

    agent_mode = utils.get_agent_mode(config)

    # Block native provider websearch in RAG mode
    if agent_mode == AgentMode.RAG and has_native_search:
        block_msg = ToolMessage(
            content="Network browsing is not allowed in RAG mode. Skipping native web search.",
            name="web_search",
            tool_call_id="native-websearch-blocked",
        )
        return Command(
            goto="compress_research", update={"researcher_messages": [block_msg]}
        )

    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")

    # Step 2: Handle other tool calls
    tools = await utils.get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=utils.apply_security_to_tool_result(
                observation, tool_call["name"], tool_call["id"], config
            ),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )
        for observation, tool_call in zip(observations, tool_calls, strict=False)
    ]

    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = (
        state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    )
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research", update={"researcher_messages": tool_outputs}
        )

    # Continue research loop with tool results
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.

    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.

    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings

    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    api_key = await utils.get_api_key_for_model(configurable.research_model, config)
    synthesizer_model = configurable_model.with_config(
        {
            "model": configurable.compression_model,
            "max_tokens": configurable.compression_model_max_tokens,
            "api_key": api_key,
            "tags": ["langsmith:nostream"],
        }
    )

    agent_mode = utils.get_agent_mode(config)

    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])

    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(
        HumanMessage(content=compress_research_simple_human_message)
    )

    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            # comments in English
            compression_prompt_template = (
                compress_research_system_prompt_rag
                if agent_mode == AgentMode.RAG
                else compress_research_system_prompt_online
            )
            compression_prompt = compression_prompt_template.format(
                date=utils.get_today_str()
            )
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages

            # Execute compression
            secured = utils.apply_security_to_model_messages(messages, config)
            response = await synthesizer_model.ainvoke(secured)
            if isinstance(response, AIMessage):
                response = utils.apply_security_to_model_output(response, config)

            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join(
                [
                    str(message.content)
                    for message in filter_messages(
                        researcher_messages, include_types=["tool", "ai"]
                    )
                ]
            )

            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
            }

        except Exception as e:
            synthesis_attempts += 1

            # Handle token limit exceeded by removing older messages
            if utils.is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = utils.remove_up_to_last_ai_message(
                    researcher_messages
                )
                continue

            # For other errors, continue retrying
            continue

    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join(
        [
            str(message.content)
            for message in filter_messages(
                researcher_messages, include_types=["tool", "ai"]
            )
        ]
    )

    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
    }


# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, output=ResearcherOutputState, config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)  # Main researcher logic
researcher_builder.add_node(
    "researcher_tools", researcher_tools
)  # Tool execution handler
researcher_builder.add_node(
    "compress_research", compress_research
)  # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")  # Entry point to researcher
researcher_builder.add_edge("compress_research", END)  # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.

    This function takes all collected research findings and synthesizes them into a
    well-structured, comprehensive final report using the configured report generation model.

    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys

    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    api_key = await utils.get_api_key_for_model(configurable.research_model, config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": api_key,
        "tags": ["langsmith:nostream"],
    }
    agent_mode = utils.get_agent_mode(config)

    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            history_messages = state.get("messages", [])
            sanitized_history_messages = utils.apply_security_to_model_messages(
                history_messages, config
            )
            history_text = get_buffer_string(sanitized_history_messages)

            safe_research_brief = utils.sanitize_text_block(
                state.get("research_brief", ""), config
            )
            safe_findings = utils.sanitize_text_block(findings, config)

            final_report_prompt_template = (
                final_report_generation_prompt_rag
                if agent_mode == AgentMode.RAG
                else final_report_generation_prompt_online
            )
            final_report_prompt = final_report_prompt_template.format(
                research_brief=safe_research_brief,
                messages=history_text,
                findings=safe_findings,
                date=utils.get_today_str(),
            )

            # Generate the final report
            secured = utils.apply_security_to_model_messages(
                [HumanMessage(content=final_report_prompt)],
                config,
            )
            final_report = await configurable_model.with_config(
                writer_model_config
            ).ainvoke(secured)
            if isinstance(final_report, AIMessage):
                final_report = utils.apply_security_to_model_output(
                    final_report, config
                )

            step_index = state.get("step_index", 0)
            configurable = Configuration.from_runnable_config(config)
            steps = configurable.steps or []

            if step_index + 1 < len(steps) and step_index != -1:
                return Command(
                    goto="provide_placeholders",
                    update={
                        **cleared_state,
                        "messages": [final_report],
                        "final_report": final_report.content,
                        "step_index": step_index + 1,
                    },
                )

            # Return successful report generation
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state,
            }

        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if utils.is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1

                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = utils.get_model_token_limit(
                        configurable.final_report_model
                    )
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [
                                AIMessage(
                                    content="Report generation failed due to token limits"
                                )
                            ],
                            **cleared_state,
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)

                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [
                        AIMessage(content="Report generation failed due to an error")
                    ],
                    **cleared_state,
                }

    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [
            AIMessage(content="Report generation failed after maximum retries")
        ],
        **cleared_state,
    }


# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, input=AgentInputState, config_schema=Configuration
)

deep_researcher_builder.add_node("before_model_mw", utils.sanitize_input)
deep_researcher_builder.add_node("provide_placeholders", provide_placeholders)
deep_researcher_builder.add_node("prepare_step", prepare_step)
deep_researcher_builder.add_node("run_sub_prompts", run_sub_prompts)
deep_researcher_builder.add_node("parallel_subprompt", parallel_subprompt)
deep_researcher_builder.add_node("fan_out_parallel", fan_out_parallel)
deep_researcher_builder.add_node("collect_parallel", collect_parallel)
deep_researcher_builder.add_node(
    "clarify_with_user", clarify_with_user
)  # User clarification phase
deep_researcher_builder.add_node(
    "write_research_brief", write_research_brief
)  # Research planning phase
deep_researcher_builder.add_node(
    "research_supervisor", supervisor_subgraph
)  # Research execution phase
deep_researcher_builder.add_node(
    "final_report_generation", final_report_generation
)  # Report generation phase

deep_researcher_builder.add_edge(START, "before_model_mw")
deep_researcher_builder.add_edge("before_model_mw", "provide_placeholders")
deep_researcher_builder.add_conditional_edges(
    "fan_out_parallel",
    lambda state: state.get("parallel_sends", []) or [],
)
deep_researcher_builder.add_edge("fan_out_parallel", "collect_parallel")
deep_researcher_builder.add_edge("parallel_subprompt", "collect_parallel")
deep_researcher_builder.add_edge("clarify_with_user", "write_research_brief")
deep_researcher_builder.add_edge("write_research_brief", "research_supervisor")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
