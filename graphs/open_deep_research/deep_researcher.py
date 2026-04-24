"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import inspect
import json
import logging
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

from graphs.open_deep_research.configuration import (
    AgentMode,
    Configuration,
)
from graphs.open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt_online_optimized,
    compress_research_system_prompt_rag,
    detail_check_prompt,
    final_report_generation_prompt_online_optimized,
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
    PromptDetailCheck,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from graphs.open_deep_research.utils import (
    anthropic_websearch_called,
    apply_placeholders,
    gemini_websearch_called,
    get_agent_mode,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    normalize_branch_name,
    normalize_placeholders,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    resolve_artifact_placeholders,
    resolve_reasoning_model_params,
    think_tool,
    truncate_result,
)

logger = logging.getLogger(__name__)


def debug_print(*args, **kwargs):
    """Debug print that uses logger.debug - visible in dev, silent in prod."""
    message = " ".join(str(arg) for arg in args)
    logger.debug(message)


def log_step_transition(from_step: str, to_step: str, reason: str = ""):
    logger.debug(f"STEP TRANSITION: {from_step} -> {to_step} | Reason: {reason}")


def log_placeholder_verification(step: str, expected: list, actual: list) -> bool:
    logger.debug(f"PLACEHOLDER VERIFY [{step}]: expected={expected}, actual={actual}")
    return True


def log_subprompt_complete(name: str, result_len: int):
    logger.debug(f"SUBPROMPT COMPLETE: {name} | result_len={result_len}")


def log_state(node_name: str, state: dict):
    keys = list(state.keys()) if state else []
    logger.debug(f"STATE [{node_name}]: keys={keys}")


def log_separator(title: str):
    logger.debug(f"{'=' * 20} {title} {'=' * 20}")


def log_step_entry(step_name: str, state: dict):
    step_index = state.get("step_index", "?")
    logger.debug(f"STEP ENTRY [{step_name}]: step_index={step_index}")


def log_step_exit(step_name: str, goto: str, updates: dict = None):
    update_keys = list(updates.keys()) if updates else []
    logger.debug(f"STEP EXIT [{step_name}]: goto={goto}, updates={update_keys}")


def log_prompt(node_name: str, prompt_type: str, content: str):
    # Truncate to avoid massive logs
    preview = content[:200] + "..." if len(content) > 200 else content
    logger.debug(f"PROMPT [{node_name}] ({prompt_type}): {preview}")


def log_response(node_name: str, content: str):
    preview = content[:200] + "..." if len(content) > 200 else content
    logger.debug(f"RESPONSE [{node_name}]: {preview}")


def log_tool_output(node_name: str, tool_name: str, output: str):
    preview = output[:150] + "..." if len(output) > 150 else output
    logger.debug(f"TOOL OUTPUT [{node_name}] ({tool_name}): {preview}")


def log_subprompt_execution(name: str, subprompt_type: str, input_text: str, result: str, result_len: int):
    input_preview = input_text[:100] + "..." if len(input_text) > 100 else input_text
    logger.debug(f"SUBPROMPT EXEC [{name}] ({subprompt_type}): input={input_preview}, result_len={result_len}")


def log_placeholder_substitution(step_name: str, template: str, placeholders: list, result: str):
    placeholder_fields = [p.get("field", "?") if isinstance(p, dict) else "?" for p in placeholders]
    logger.debug(
        f"PLACEHOLDER SUB [{step_name}]: fields={placeholder_fields}, template_len={len(template)}, result_len={len(result)}"
    )


def log_synthetic_placeholders(placeholders: list, phase: str):
    fields = [p.get("field", "?") if isinstance(p, dict) else "?" for p in placeholders]
    logger.debug(f"SYNTHETIC PLACEHOLDERS [{phase}]: count={len(placeholders)}, fields={fields}")


def log_parallel_merge(step_name: str, result_map: dict, before: list, after: list):
    logger.debug(
        f"PARALLEL MERGE [{step_name}]: result_keys={list(result_map.keys())}, before={len(before)}, after={len(after)}"
    )


def log_data_sizes(step_name: str, data_dict: dict):
    sizes = {k: len(v) if hasattr(v, "__len__") else "?" for k, v in data_dict.items()}
    logger.debug(f"DATA SIZES [{step_name}]: {sizes}")


def log_messages_trace(node_name: str, messages: list):
    types = [type(m).__name__ for m in messages] if messages else []
    logger.debug(f"MESSAGES TRACE [{node_name}]: count={len(messages)}, types={types}")


def log_state_full(node_name: str, state: dict):
    logger.debug(f"STATE FULL [{node_name}]: {state}")


def log_final_report_assembly(brief_raw: str, brief_after: str, findings: str, placeholders: dict):
    placeholder_count = len(placeholders) if placeholders else 0
    logger.debug(
        f"FINAL REPORT ASSEMBLY: brief_raw_len={len(brief_raw)}, brief_after_len={len(brief_after)}, findings_len={len(findings)}, placeholders={placeholder_count}"
    )


def log_error(node_name: str, error: Exception):
    logger.debug(f"ERROR [{node_name}]: {type(error).__name__}: {error}")


# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "reasoning_effort", "model_kwargs"),
)


def parse_start_message(message_text: str) -> tuple[bool, list | None]:
    """Parse a message starting with 'Start' followed by optional JSON placeholders.

    Expected format: Start [{"field_name1": "value1", "field_name2": "value2"}, ...]
    Each element in the array corresponds to placeholders for each step (by step_index).

    Returns:
        tuple: (is_start_message, all_steps_placeholders or None)
    """
    if not message_text or not message_text.strip().startswith("Start"):
        return False, None

    text = message_text.strip()

    # Check if it's just "Start" with no placeholders
    if text == "Start":
        return True, []

    # Try to extract JSON array after "Start"
    json_part = text[5:].strip()  # Remove "Start" prefix
    if not json_part:
        return True, []

    try:
        placeholders = json.loads(json_part)
        if isinstance(placeholders, list):
            return True, placeholders
        return True, []
    except json.JSONDecodeError:
        return True, []


async def provide_placeholders(state: AgentState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    steps = configurable.steps or []

    last_message = state.get("messages", [])[-1]

    if not steps:
        return Command(goto="prepare_step")

    # Parse the message to check for "Start" format with inline placeholders
    message_text = ""
    if isinstance(last_message, HumanMessage):
        message_text = last_message.text() if hasattr(last_message, "text") else str(last_message.content)

    is_start_message, all_steps_placeholders = parse_start_message(message_text)

    if isinstance(last_message, HumanMessage) and not is_start_message:
        return Command(goto="clarify_with_user", update={"step_index": -1})

    step_index = state.get("step_index", 0)
    current_placeholders = state.get("placeholders", [])

    # Extract field names for debugging
    current_fields = [
        p.get("field") if isinstance(p, dict) else (p.field if hasattr(p, "field") else "?")
        for p in current_placeholders
    ]

    debug_print(
        f"provide_placeholders ENTRY: step_index={step_index}, "
        f"current_placeholders={len(current_placeholders)}, "
        f"fields={current_fields}"
    )

    if step_index >= len(steps):
        return Command(goto=END)

    current_step = steps[step_index]

    has_sub_prompts = bool(current_step.parallel_sub_prompts or current_step.sequential_sub_prompts)
    goto = "run_sub_prompts" if has_sub_prompts else "prepare_step"

    expected_placeholders = [x.strip() for x in current_step.placeholder_names if x.strip()]
    expected_placeholders = list(dict.fromkeys(expected_placeholders))

    debug_print(f"provide_placeholders: step={step_index}, expected_placeholders={expected_placeholders}")

    if not expected_placeholders:
        return Command(goto=goto)

    # Check if we have inline placeholders from the "Start" message
    # Format: Start [{"company_name": "Acme", "context": "..."}, {"other_field": "value"}]
    # Each array element corresponds to a step by index
    if all_steps_placeholders and step_index < len(all_steps_placeholders):
        step_placeholders_dict = all_steps_placeholders[step_index]

        if isinstance(step_placeholders_dict, dict):
            # Check if all expected placeholders are provided for this step
            all_provided = all(field in step_placeholders_dict for field in expected_placeholders)

            if all_provided:
                # Convert dict format to list of {"field": key, "value": val} for normalize_placeholders
                placeholders_list = [{"field": key, "value": value} for key, value in step_placeholders_dict.items()]
                debug_print(
                    f"provide_placeholders: step={step_index}, "
                    f"raw placeholders_list={[{k: (v[:80] + '...' if isinstance(v, str) and len(v) > 80 else v) for k, v in p.items()} for p in placeholders_list]}"
                )
                normalized = normalize_placeholders(placeholders_list)
                debug_print(
                    f"provide_placeholders: step={step_index}, "
                    f"after normalize: {[{k: (v[:80] + '...' if isinstance(v, str) and len(v) > 80 else v) for k, v in p.items()} for p in normalized]}"
                )
                # Resolve artifact-type values if present (no-op for regular string values)
                normalized = await resolve_artifact_placeholders(normalized, config)
                debug_print(
                    f"provide_placeholders EXIT (inline): step={step_index}, "
                    f"returning {len(normalized)} placeholders, "
                    f"fields={[p.get('field') for p in normalized if isinstance(p, dict)]}"
                )
                # Use the inline placeholders directly, no interrupt needed
                # Use standard append logic (no "override") to preserve history
                return Command(
                    goto=goto,
                    update={
                        "placeholders": normalized,
                    },
                )

    # Check if current state already has all expected placeholders
    current_placeholder_fields = {
        p.get("field") if isinstance(p, dict) else (p.field if hasattr(p, "field") else None)
        for p in current_placeholders
    }
    current_placeholder_fields.discard(None)

    # Merge synthetic placeholders into the check
    # These are results from previous steps (e.g. Step_1__Company_Intelligence)
    # that act as valid placeholders for the current step
    synthetic_placeholders = state.get("synthetic_placeholders", []) or []
    for sp in synthetic_placeholders:
        if isinstance(sp, dict) and sp.get("field"):
            current_placeholder_fields.add(sp.get("field"))

    missing_placeholders = [field for field in expected_placeholders if field not in current_placeholder_fields]

    if not missing_placeholders:
        # All expected placeholders already present in state - no need to ask again
        debug_print(
            f"provide_placeholders EXIT (from state): step={step_index}, "
            f"all {len(expected_placeholders)} expected placeholders already present"
        )
        return Command(goto=goto)

    # Fall back to interrupt if placeholders not provided inline
    debug_print(f"provide_placeholders: step={step_index}, missing placeholders: {missing_placeholders}")
    human_payload = interrupt(
        {
            "type": "placeholders_required",
            "expected": expected_placeholders,
            "step_index": step_index,
        }
    )

    if isinstance(human_payload, dict) and human_payload.get("skip"):
        debug_print(f"provide_placeholders EXIT (skip): step={step_index}")
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
                "placeholders": [],  # Append nothing (preserve history)
                "step_index": step_index + 1,
            },
        )

    normalized = normalize_placeholders(human_payload)
    debug_print(f"provide_placeholders EXIT (user input): step={step_index}, returning {len(normalized)} placeholders")
    return Command(
        goto=goto,
        update={
            "placeholders": {
                "type": "override",
                "value": normalized,
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

    # LOG: Track state at entry for debugging
    log_state("run_sub_prompts", state)
    debug_print(f"run_sub_prompts ENTER: step={step_index}, phase={phase}")

    if phase == "announce":
        # Build deterministic metadata for both kinds of sub-prompts
        used: set[str] = set()
        seq_meta: list[dict] = []
        par_meta: list[dict] = []
        ui_messages: list[ToolMessage] = []

        for it in step.sequential_sub_prompts or []:
            name = normalize_branch_name(it.name, used)
            preview = apply_placeholders(it.text, user_placeholders)
            ui_messages.append(
                ToolMessage(
                    content=f"Starting sequential subprompt: {name}\nTopic: {truncate_result(preview)}",
                    name="SequentialSubprompt",
                    tool_call_id=f"sequential:{name}:start",
                )
            )
            seq_meta.append({"name": name, "text_template": it.text})

        for it in step.parallel_sub_prompts or []:
            name = normalize_branch_name(it.name, used)
            preview = apply_placeholders(it.text, user_placeholders)
            ui_messages.append(
                ToolMessage(
                    content=f"Starting parallel subprompt: {name}\nTopic: {truncate_result(preview)}",
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
        debug_print(
            f"run_sub_prompts SEQUENTIAL '{name}': "
            f"user={len(user_placeholders)}, synthetic={len(synthetic_placeholders)}, merged={len(merged_ph)}"
        )

        # Log placeholder substitution for sequential subprompt
        base_text = apply_placeholders(text, merged_ph)
        log_placeholder_substitution(f"SEQUENTIAL:{name}", text, merged_ph, base_text)

        ctx_text = ("\n\nContext from previous steps:\n" + "\n\n".join(context_chunks)) if context_chunks else ""
        final_text = base_text + ctx_text

        # Log the complete input to researcher
        log_subprompt_execution(name, "sequential", final_text, "[EXECUTING...]", len(final_text))

        try:
            observation = await researcher_subgraph.ainvoke(
                {
                    "researcher_messages": [HumanMessage(content=final_text)],
                    "research_topic": final_text,
                },
                config,
            )
            compressed = observation.get("compressed_research", "")
            # Log successful result
            log_subprompt_execution(name, "sequential", final_text, compressed, len(compressed))
        except Exception as e:
            import traceback

            traceback.print_exc()
            log_error(f"SEQUENTIAL:{name}", e)
            compressed = f"Error during sequential subprompt '{name}': {e}"

        context_chunks.append(compressed)
        ui_messages.append(
            ToolMessage(
                content=f"Completed: {name}\nShort result:\n{truncate_result(compressed)}",
                name="SequentialSubprompt",
                tool_call_id=f"sequential:{name}:done",
            )
        )
        synthetic_placeholders.append({"field": name, "value": compressed})
        log_subprompt_complete(name, len(compressed))

        # Log synthetic placeholder state after each sequential step
        log_synthetic_placeholders(synthetic_placeholders, f"after_sequential_{name}")

    # Log accumulated synthetic placeholders after sequential execution
    log_synthetic_placeholders(synthetic_placeholders, "after_all_sequential")

    # 2) Execute parallel
    parallel_sends: list[Send] = []

    merged_for_parallel = user_placeholders + synthetic_placeholders
    debug_print(
        f"run_sub_prompts PARALLEL: "
        f"user={len(user_placeholders)}, synthetic={len(synthetic_placeholders)}, "
        f"merged={len(merged_for_parallel)}, parallel_count={len(par_meta)}"
    )
    for item in par_meta:
        name = item.get("name", "")
        text = item.get("text_template", "")
        if not name:
            continue
        parallel_sends.append(
            {
                "subprompt_name": name,
                "subprompt_template": text,
                "placeholders": {"type": "override", "value": merged_for_parallel},
            }
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


def route_parallel_execution(state: AgentState) -> list[Send]:
    """Route to parallel subprompts based on state."""
    try:
        parallel_sends = state.get("parallel_sends", []) or []
        debug_print(f"route_parallel_execution: sends={len(parallel_sends)}")

        sends = []
        for send_conf in parallel_sends:
            sends.append(Send("parallel_subprompt", send_conf))

        return sends
    except Exception as e:
        logger.error(f"Error in route_parallel_execution: {e}")
        # Return empty list on error to avoid crash, though this might stall the graph
        return []


async def parallel_subprompt(state: AgentState, config: RunnableConfig):
    """Worker for a single parallel subprompt (one subprompt per node call)."""
    nm = state.get("subprompt_name", "subprompt")
    tmpl = state.get("subprompt_template", "")
    placeholders = state.get("placeholders", []) or []

    if isinstance(placeholders, dict) and placeholders.get("type") == "override":
        placeholders = placeholders.get("value", [])

    debug_print(f"parallel_subprompt '{nm}': placeholders={len(placeholders)}")

    # Log placeholder substitution for parallel subprompt
    sub_text = apply_placeholders(tmpl, placeholders)
    log_placeholder_substitution(f"PARALLEL:{nm}", tmpl, placeholders, sub_text)

    # Log the input before execution
    log_subprompt_execution(nm, "parallel", sub_text, "[EXECUTING...]", len(sub_text))

    comp = ""
    try:
        observation = await researcher_subgraph.ainvoke(
            {
                "researcher_messages": [HumanMessage(content=sub_text)],
                "research_topic": sub_text,
            },
            config,
        )
        comp = observation.get("compressed_research", "")
        # Log successful result
        log_subprompt_execution(nm, "parallel", sub_text, comp, len(comp))
    except Exception as e:
        # Handle token limit exceeded by truncating content and retrying
        log_error(f"PARALLEL:{nm}", e)
        configurable = Configuration.from_runnable_config(config)
        if is_token_limit_exceeded(e, configurable.research_model):
            # Use dynamic token limit based on model, with 10% reduction on each retry
            model_token_limit = get_model_token_limit(configurable.research_model)
            if model_token_limit:
                char_limit = model_token_limit * 4
            else:
                # Fallback to a conservative default if model not in lookup table
                char_limit = 120000

            max_retries = 3
            current_retry = 0

            while current_retry < max_retries:
                try:
                    debug_print(
                        f"parallel_subprompt '{nm}': Token limit exceeded, retrying with char_limit={char_limit}"
                    )
                    truncated_text = truncate_result(sub_text, char_limit)
                    observation = await researcher_subgraph.ainvoke(
                        {
                            "researcher_messages": [HumanMessage(content=truncated_text)],
                            "research_topic": truncated_text,
                        },
                        config,
                    )
                    comp = observation.get("compressed_research", "")
                    log_subprompt_execution(nm, "parallel", truncated_text, comp, len(comp))
                    break
                except Exception as e2:
                    current_retry += 1
                    if is_token_limit_exceeded(e2, configurable.research_model):
                        # Reduce by 10% and retry
                        char_limit = int(char_limit * 0.9)
                        debug_print(f"parallel_subprompt '{nm}': Still exceeding limit, reducing to {char_limit}")
                        continue
                    else:
                        log_error(f"PARALLEL:{nm}:RETRY", e2)
                        comp = f"Error: Parallel sub-prompt failed during retry. {e2}"
                        break
            else:
                # All retries exhausted
                log_error(f"PARALLEL:{nm}:MAX_RETRIES", e)
                comp = f"Error: Parallel sub-prompt failed after {max_retries} retries due to context limit."
        else:
            comp = f"Error during parallel subprompt '{nm}': {e}"

    log_subprompt_complete(nm, len(comp))

    ui_msg = ToolMessage(
        content=f"Completed: {nm}\nShort result:\n{truncate_result(comp)}",
        name="ParallelSubprompt",
        tool_call_id=f"parallel:{nm}:done",
    )

    return {
        "messages": [ui_msg],
        "last_parallel_result": [{"name": nm, "compressed": comp}],
    }


async def collect_parallel(state: AgentState, config: RunnableConfig):
    """Accumulate results from parallel_subprompt workers; continue when all done."""
    log_separator("COLLECT_PARALLEL")

    result_map: dict = state.get("subprompt_results", {}) or {}
    pending: int = state.get("pending_parallel", 0) or 0

    batch: list = state.get("last_parallel_result", []) or []
    debug_print(
        f"collect_parallel ENTRY: pending={pending}, batch_size={len(batch)}, result_map_keys={list(result_map.keys())}"
    )

    for item in batch:
        nm = item.get("name")
        comp = item.get("compressed")
        if nm is not None:
            debug_print(f"collect_parallel: Adding result for '{nm}' ({len(comp) if comp else 0} chars)")
            result_map[nm] = comp
            pending = max(0, pending - 1)

    if pending > 0:
        debug_print(f"collect_parallel: Still waiting for {pending} parallel results")
        return {
            "subprompt_results": result_map,
            "pending_parallel": pending,
            "last_parallel_result": {"type": "override", "value": []},
        }

    debug_print("collect_parallel: All parallel results collected, proceeding to merge")

    synthetic_placeholders_before: list = list(state.get("synthetic_placeholders", []) or [])
    synthetic_placeholders: list = state.get("synthetic_placeholders", []) or []
    user_placeholders: list = state.get("placeholders", []) or []
    existing = {p["field"] for p in user_placeholders if isinstance(p, dict) and "field" in p}

    for k, v in result_map.items():
        if k not in existing:
            synthetic_placeholders.append({"field": k, "value": v})

    merged_placeholders = user_placeholders + [sp for sp in synthetic_placeholders if sp.get("field") not in existing]

    # Log the merge result with detailed tracking
    debug_print(f"collect_parallel: merged {len(result_map)} parallel results")
    log_parallel_merge("collect_parallel", result_map, synthetic_placeholders_before, synthetic_placeholders)
    log_synthetic_placeholders(synthetic_placeholders, "after_parallel_merge")

    # Log data sizes for monitoring
    log_data_sizes(
        "collect_parallel",
        {
            "result_map": result_map,
            "user_placeholders": user_placeholders,
            "synthetic_placeholders": synthetic_placeholders,
            "merged_placeholders": merged_placeholders,
        },
    )

    return Command(
        goto="prepare_step",
        update={
            "subprompt_results": result_map,
            "placeholders": {"type": "override", "value": merged_placeholders},
            "parallel_sends": None,
            "pending_parallel": 0,
            "last_parallel_result": {"type": "override", "value": []},
            # CRITICAL FIX: Do NOT clear synthetic_placeholders - they must accumulate
            # between steps for final assembly to use step results as placeholders
            "synthetic_placeholders": synthetic_placeholders,
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
    current_placeholders = state.get("placeholders", [])

    # CRITICAL FIX: Merge synthetic placeholders for prompt application
    # This ensures findings from previous steps are injected into the current prompt
    synthetic_placeholders = state.get("synthetic_placeholders", []) or []

    merged_placeholders_map = {}

    for sp in synthetic_placeholders:
        if isinstance(sp, dict) and sp.get("field"):
            merged_placeholders_map[sp.get("field")] = sp

    for p in current_placeholders:
        if isinstance(p, dict) and p.get("field"):
            merged_placeholders_map[p.get("field")] = p

    all_placeholders = list(merged_placeholders_map.values())

    prompt_text = apply_placeholders(step.text, all_placeholders)

    log_state("prepare_step", state)
    debug_print(
        f"prepare_step: step={step_index}, "
        f"placeholders={len(current_placeholders)}, "
        f"text_len={len(step.text)} → {len(prompt_text)}"
    )

    # Logic to detect "Assembly" steps (steps with no sub-prompts that are populated with findings)
    # If the prompt is large (likely containing findings) and has no sub-prompts,
    # we skip the supervisor/researcher loop and go straight to report generation.
    has_sub_prompts = bool(step.sequential_sub_prompts or step.parallel_sub_prompts)
    is_large_context_step = len(prompt_text) > 500  # Heuristic: >500 chars usually means data injection

    # Check if we're returning AFTER sub-prompts have completed
    # If sequential_context has data, it means sub-prompts ran and we should go to report generation
    sequential_context = state.get("sequential_context", []) or []
    subprompt_results = state.get("subprompt_results", {}) or {}
    has_completed_subprompts = bool(sequential_context) or bool(subprompt_results)

    debug_print(
        f"prepare_step: has_sub_prompts={has_sub_prompts}, is_large_context_step={is_large_context_step}, has_completed_subprompts={has_completed_subprompts}"
    )

    # CASE 1: Assembly step (no sub-prompts, large context)
    if not has_sub_prompts and is_large_context_step:
        debug_print(
            f"prepare_step: Detecting Assembly Step -> Routing directly to final_report_generation. Step Index: {step_index}"
        )
        return Command(
            goto="final_report_generation",
            update={
                "research_brief": {"type": "override", "value": prompt_text},
                "step_index": step_index,
                "notes": {"type": "override", "value": []},
                "sequential_context": {"type": "override", "value": []},
                "subprompt_results": {"type": "override", "value": {}},
            },
        )

    # CASE 2: Sub-prompts completed - go directly to final_report_generation
    # This prevents the supervisor from re-researching what sub-prompts already collected
    if has_completed_subprompts:
        debug_print(
            f"prepare_step: Sub-prompts completed -> Routing directly to final_report_generation. Step Index: {step_index}"
        )
        return Command(
            goto="final_report_generation",
            update={
                "research_brief": {"type": "override", "value": prompt_text},
                "step_index": step_index,
                # Keep the collected data from sub-prompts
            },
        )

    # CASE 3: Normal step - go to clarify_with_user -> write_research_brief -> supervisor
    debug_print(f"prepare_step: Routing to clarify_with_user. Step Index: {step_index}")
    return Command(
        goto="clarify_with_user",
        update={
            "messages": [HumanMessage(content=prompt_text)],
            "step_index": step_index,
            "research_brief": {"type": "override", "value": prompt_text},  # Save step.text for later
            # Placeholders preserved - no override to maintain context across steps
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
    api_key = await get_api_key_for_model(configurable.research_model, config)
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": api_key,
        "tags": ["langsmith:nostream"],
        **resolve_reasoning_model_params(
            configurable.research_model,
            configurable.research_model_reasoning_level.value
            if configurable.research_model_reasoning_level is not None
            else None,
        ),
    }

    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model.with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(messages=get_buffer_string(messages), date=get_today_str())
    log_messages_trace("clarify_with_user", messages)
    log_prompt("clarify_with_user", "PROMPT", prompt_content)

    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    log_response("clarify_with_user", str(response))

    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]},
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
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
    api_key = await get_api_key_for_model(configurable.research_model, config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": api_key,
        "tags": ["langsmith:nostream"],
        **resolve_reasoning_model_params(
            configurable.research_model,
            configurable.research_model_reasoning_level.value
            if configurable.research_model_reasoning_level is not None
            else None,
        ),
    }

    # Configure model for structured research question generation
    research_model = (
        configurable_model.with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])), date=get_today_str()
    )
    log_prompt("write_research_brief", "PROMPT", prompt_content)

    # CHECK: Use LLM to determine if the prompt is detailed enough to use directly
    # This replaces flaky keyword matching with intelligent analysis
    messages = state.get("messages", [])
    last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    use_raw_prompt = False
    if last_human_msg and len(last_human_msg.content) > 200:  # Only check longer prompts
        try:
            # Configure model for prompt detail check
            detail_check_model = (
                configurable_model.with_structured_output(PromptDetailCheck)
                .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
                .with_config(research_model_config)
            )

            debug_print(f"write_research_brief: Checking detail of prompt: {last_human_msg.content[:50]}...")

            formatted_check_prompt = detail_check_prompt.format(user_prompt=last_human_msg.content)

            detail_check = await detail_check_model.ainvoke([HumanMessage(content=formatted_check_prompt)])
            use_raw_prompt = detail_check.is_detailed_enough
            debug_print(
                f"write_research_brief: LLM detail check - is_detailed_enough={use_raw_prompt}, reasoning={detail_check.reasoning}"
            )
        except Exception as e:
            debug_print(f"write_research_brief: Detail check failed, falling back to rewrite: {e}")
            use_raw_prompt = False

    if use_raw_prompt and last_human_msg:
        # BYPASS: Use the provided prompt directly as the research brief
        debug_print("write_research_brief: Detailed prompt detected by LLM. Bypassing Rewrite.")
        response = ResearchQuestion(
            research_brief=last_human_msg.content,
        )
        log_response("write_research_brief", "BYPASS: Using raw user prompt (LLM confirmed detailed).")
    else:
        # NORMAL: Rewrite the prompt into a research brief
        response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
        log_response("write_research_brief", str(response))

    # Step 3: Initialize supervisor with research brief and instructions
    agent_mode = get_agent_mode(config)
    supervisor_prompt_template = (
        lead_researcher_prompt_rag if agent_mode == AgentMode.RAG else lead_researcher_prompt_online
    )

    supervisor_system_prompt = supervisor_prompt_template.format(
        date=get_today_str(),
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


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
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
    api_key = await get_api_key_for_model(configurable.research_model, config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": api_key,
        "tags": ["langsmith:nostream"],
        **resolve_reasoning_model_params(
            configurable.research_model,
            configurable.research_model_reasoning_level.value
            if configurable.research_model_reasoning_level is not None
            else None,
        ),
    }

    agent_mode = get_agent_mode(config)

    # Choose base tools by mode: ConductResearch only in ONLINE
    if agent_mode == AgentMode.RAG:
        base_tools = [ResearchComplete, think_tool]
    else:
        base_tools = [ConductResearch, ResearchComplete, think_tool]

    dynamic_tools = await get_all_tools(config)
    supervisor_dynamic = [t for t in dynamic_tools if not isinstance(t, dict)]
    lead_researcher_tools = base_tools + supervisor_dynamic

    seen_names: set[str] = set()
    unique_tools = []
    for t in lead_researcher_tools:
        name = getattr(t, "name", None) or getattr(t, "__name__", None) or str(t)
        if name not in seen_names:
            seen_names.add(name)
            unique_tools.append(t)
    lead_researcher_tools = unique_tools

    native_search_filtered = len(dynamic_tools) - len(supervisor_dynamic)
    logger.debug(
        "supervisor: Binding %d tools (base=%d, dynamic=%d, native_search_filtered=%d). "
        "search_api=%s, model=%s, tool_names=%s",
        len(lead_researcher_tools),
        len(base_tools),
        len(supervisor_dynamic),
        native_search_filtered,
        configurable.search_api.value,
        configurable.research_model,
        [getattr(t, "name", None) or str(t) for t in lead_researcher_tools],
    )

    # Configure model with tools, retry logic, and model settings.
    # tool_choice="any" forces the model to call at least one tool.
    research_model = (
        configurable_model.bind_tools(lead_researcher_tools, tool_choice="any")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])

    log_messages_trace("supervisor", supervisor_messages)
    log_prompt("supervisor", "MESSAGES_CONTEXT", get_buffer_string(supervisor_messages))

    response = await research_model.ainvoke(supervisor_messages)

    log_response("supervisor", str(response))

    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
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

    rag_mode = get_agent_mode(config) == AgentMode.RAG

    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls
    )

    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
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
    think_tool_calls = [tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "think_tool"]

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
        tools = await get_all_tools(config)
        tools_by_name = {
            (getattr(t, "name", None) or t.get("name")): t for t in tools if (getattr(t, "name", None) or t.get("name"))
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
                log_tool_output("supervisor", name, str(tool_content))

                if name == "rag_search":
                    tool_content = json.dumps(
                        {"answer": result, "question": args.get("query")},
                        ensure_ascii=False,
                    )

                all_tool_messages.append(
                    ToolMessage(
                        content=tool_content,
                        name=name,
                        tool_call_id=tc["id"],
                        additional_kwargs={"visibility": "internal"},
                    )
                )

    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "ConductResearch"
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
                allowed_conduct_research_calls = conduct_research_calls[: configurable.max_concurrent_research_units]
                overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units :]

                # Execute research tasks in parallel
                research_tasks = [
                    researcher_subgraph.ainvoke(
                        {
                            "researcher_messages": [HumanMessage(content=tool_call["args"]["research_topic"])],
                            "research_topic": tool_call["args"]["research_topic"],
                        },
                        config,
                    )
                    for tool_call in allowed_conduct_research_calls
                ]

                tool_results = await asyncio.gather(*research_tasks)

                # Create tool messages with research results
                for observation, tool_call in zip(tool_results, allowed_conduct_research_calls, strict=False):
                    all_tool_messages.append(
                        ToolMessage(
                            content=observation.get(
                                "compressed_research",
                                "Error synthesizing research report: Maximum retries exceeded",
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
                    ["\n".join(observation.get("raw_notes", [])) for observation in tool_results]
                )

                if raw_notes_concat:
                    update_payload["raw_notes"] = [raw_notes_concat]

            except Exception as e:
                # Handle research execution errors
                if is_token_limit_exceeded(e, configurable.research_model):
                    # Token limit exceeded - end research phase
                    return Command(
                        goto=END,
                        update={
                            "notes": get_notes_from_tool_calls(supervisor_messages),
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
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
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
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError("No tools found to conduct research: Please configure your search API")

    # Step 2: Configure the researcher model with tools
    api_key = await get_api_key_for_model(configurable.research_model, config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": api_key,
        "tags": ["langsmith:nostream"],
        **resolve_reasoning_model_params(
            configurable.research_model,
            configurable.research_model_reasoning_level.value
            if configurable.research_model_reasoning_level is not None
            else None,
        ),
    }

    # Choose mode-specific researcher prompt
    agent_mode = get_agent_mode(config)
    prompt_template = research_system_prompt_rag if agent_mode == AgentMode.RAG else research_system_prompt_online
    researcher_prompt = prompt_template.format(
        date=get_today_str(),
        search_tool=configurable.search_api.value,
    )

    logger.debug(
        "researcher: Binding %d tools. search_api=%s, model=%s, tool_names=%s",
        len(tools),
        configurable.search_api.value,
        configurable.research_model,
        [getattr(t, "name", None) or str(t) for t in tools],
    )

    # Configure model with tools, retry logic, and settings.
    # tool_choice="any" forces the model to call at least one tool rather than
    # responding with plain text (some models, e.g. gpt-5-mini, skip tool calls otherwise).
    research_model = (
        configurable_model.bind_tools(tools, tool_choice="any")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)]
    if configurable.sales_context_prompt:
        messages.append(SystemMessage(content=configurable.sales_context_prompt))
    messages += researcher_messages

    log_messages_trace("researcher", messages)
    log_prompt("researcher", "FULL_PROMPT", get_buffer_string(messages))

    response = await research_model.ainvoke(messages)

    log_response("researcher", str(response))

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
    has_native_search = (
        openai_websearch_called(most_recent_message)
        or anthropic_websearch_called(most_recent_message)
        or gemini_websearch_called(most_recent_message)
    )

    agent_mode = get_agent_mode(config)

    # Block native provider websearch in RAG mode
    if agent_mode == AgentMode.RAG and has_native_search:
        block_msg = ToolMessage(
            content="Network browsing is not allowed in RAG mode. Skipping native web search.",
            name="web_search",
            tool_call_id="native-websearch-blocked",
        )
        return Command(goto="compress_research", update={"researcher_messages": [block_msg]})

    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")

    if has_native_search and not has_tool_calls:
        logger.debug(
            "researcher_tools: Native search detected without tool_calls. "
            "Results are in model response. iteration=%d/%d",
            state.get("tool_call_iterations", 0),
            configurable.max_react_tool_calls,
        )
        exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
        if exceeded_iterations:
            return Command(goto="compress_research")
        return Command(goto="researcher")

    # Step 2: Handle other tool calls
    tools = await get_all_tools(config)
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool for tool in tools}

    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # Log raw observations for hyper-logging
    for obs, tc in zip(observations, tool_calls, strict=False):
        log_tool_output("researcher", tc["name"], str(obs))

    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(content=observation, name=tool_call["name"], tool_call_id=tool_call["id"])
        for observation, tool_call in zip(observations, tool_calls, strict=False)
    ]

    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(goto="compress_research", update={"researcher_messages": tool_outputs})

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
    api_key = await get_api_key_for_model(configurable.research_model, config)
    synthesizer_model = configurable_model.with_config(
        {
            "model": configurable.compression_model,
            "max_tokens": configurable.compression_model_max_tokens,
            "api_key": api_key,
            "tags": ["langsmith:nostream"],
            **resolve_reasoning_model_params(
                configurable.compression_model,
                configurable.compression_model_reasoning_level.value
                if configurable.compression_model_reasoning_level is not None
                else None,
            ),
        }
    )

    agent_mode = get_agent_mode(config)

    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])

    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

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
                else compress_research_system_prompt_online_optimized
            )
            compression_prompt = compression_prompt_template.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages

            log_prompt("compress_research", "PROMPT", get_buffer_string(messages))

            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            log_response("compress_research", str(response.content))

            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join(
                [str(message.content) for message in filter_messages(researcher_messages, include_types=["tool", "ai"])]
            )

            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
            }

        except Exception as e:
            synthesis_attempts += 1

            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            # For other errors, continue retrying
            continue

    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join(
        [str(message.content) for message in filter_messages(researcher_messages, include_types=["tool", "ai"])]
    )

    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
    }


# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)  # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)  # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)  # Research compression

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
    log_separator("FINAL_REPORT_GENERATION")
    log_step_entry("final_report_generation", dict(state))

    # Step 1: Extract research findings
    notes = state.get("notes", [])
    sequential_context = state.get("sequential_context", []) or []
    subprompt_results = state.get("subprompt_results", {}) or {}
    synthetic_placeholders = state.get("synthetic_placeholders", []) or []

    debug_print(
        f"final_report_generation ENTRY: notes={len(notes)}, "
        f"sequential_context={len(sequential_context)}, "
        f"subprompt_results={len(subprompt_results)}, "
        f"synthetic_placeholders={len(synthetic_placeholders)}"
    )

    # Log all synthetic placeholders in detail
    log_synthetic_placeholders(synthetic_placeholders, "final_report_entry")

    configurable = Configuration.from_runnable_config(config)
    steps = configurable.steps or []
    step_index = state.get("step_index", 0)
    current_step = steps[step_index] if step_index < len(steps) else None

    synthetic_placeholder_names = {sp.get("field") for sp in synthetic_placeholders if isinstance(sp, dict)}

    is_assembly_step = False
    if current_step and current_step.placeholders and synthetic_placeholder_names:
        is_assembly_step = bool(set(current_step.placeholder_names) & synthetic_placeholder_names)

    debug_print(
        f"final_report_generation: is_assembly_step={is_assembly_step} (step {step_index}: expecting {current_step.placeholder_names if current_step else []}, have synthetic: {list(synthetic_placeholder_names)})"
    )

    debug_print(
        f"final_report_generation: is_assembly_step={is_assembly_step} (step {step_index}: expecting {current_step.placeholders if current_step else []}, have synthetic: {list(synthetic_placeholder_names)})"
    )

    # Build findings based on step type
    findings_parts = []

    if notes:
        findings_parts.append("# Research Notes from Supervisor\n\n" + "\n\n".join(notes))

    # For assembly step: data goes via placeholder substitution, not findings
    # For regular step: add sequential and parallel results to findings
    if not is_assembly_step:
        if sequential_context:
            findings_parts.append("# Sequential Sub-Prompt Results\n\n" + "\n\n---\n\n".join(sequential_context))

        if subprompt_results:
            parallel_results = "\n\n".join([f"## {name}\n\n{content}" for name, content in subprompt_results.items()])
            findings_parts.append("# Parallel Sub-Prompt Results\n\n" + parallel_results)

    findings = "\n\n---\n\n".join(findings_parts) if findings_parts else ""

    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    api_key = await get_api_key_for_model(configurable.research_model, config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": api_key,
        "tags": ["langsmith:nostream"],
        **resolve_reasoning_model_params(
            configurable.final_report_model,
            configurable.final_report_model_reasoning_level.value
            if configurable.final_report_model_reasoning_level is not None
            else None,
        ),
    }
    agent_mode = get_agent_mode(config)

    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt_template = (
                final_report_generation_prompt_rag
                if agent_mode == AgentMode.RAG
                else final_report_generation_prompt_online_optimized
            )
            # Extract research brief value cleanly (handling potential dict from reducer)
            research_brief_val = state.get("research_brief", "")
            if isinstance(research_brief_val, dict) and research_brief_val.get("type") == "override":
                research_brief_val = research_brief_val.get("value", "")

            # Apply placeholders to research_brief (for assembly steps)
            # This substitutes [Step_1__Company_Intelligence] with actual research data
            placeholders = state.get("placeholders", []) or []

            # Build all_placeholders avoiding duplicates
            all_placeholders = list(placeholders)
            existing_fields = {p.get("field") for p in all_placeholders if isinstance(p, dict) and p.get("field")}

            # Add synthetic_placeholders (contains sequential + merged parallel from previous steps)
            for sp in synthetic_placeholders:
                field = sp.get("field") if isinstance(sp, dict) else None
                if field and field not in existing_fields:
                    all_placeholders.append(sp)
                    existing_fields.add(field)

            # Add current step's subprompt_results
            for name, content in subprompt_results.items():
                if name not in existing_fields:
                    all_placeholders.append({"field": name, "value": content})
                    existing_fields.add(name)

            # CRITICAL: Log all placeholders before substitution
            debug_print(f"final_report_generation: all_placeholders count = {len(all_placeholders)}")
            debug_print(f"final_report_generation: placeholder fields = {list(existing_fields)}")

            research_brief_before = research_brief_val

            if all_placeholders:
                research_brief_val = apply_placeholders(research_brief_val, all_placeholders)

            # Log the complete placeholder substitution details
            log_final_report_assembly(research_brief_before, research_brief_val, findings, all_placeholders)

            final_report_prompt = final_report_prompt_template.format(
                research_brief=research_brief_val,
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str(),
            )

            # Log the prompt being sent to LLM
            log_prompt("final_report_generation", "FINAL_REPORT_PROMPT", final_report_prompt)

            # Generate the final report
            final_report_messages = []
            if configurable.system_prompt:
                final_report_messages.append(SystemMessage(content=configurable.system_prompt))
            if configurable.sales_context_prompt:
                final_report_messages.append(SystemMessage(content=configurable.sales_context_prompt))
            final_report_messages.append(HumanMessage(content=final_report_prompt))

            final_report = await configurable_model.with_config(writer_model_config).ainvoke(final_report_messages)

            # Log the LLM response
            log_response("final_report_generation", str(final_report.content))

            step_index = state.get("step_index", 0)
            configurable = Configuration.from_runnable_config(config)
            steps = configurable.steps or []

            if step_index + 1 < len(steps) and step_index != -1:
                # Transitioning to next step

                # CRITICAL FIX: Preserve ALL subprompt results for assembly step
                # synthetic_placeholders already contains sequential results (Step_1__, Step_2__, Step_3__)
                # subprompt_results contains parallel results (Step_4__, Step_5__, etc.)
                # We need to merge them for the assembly step to have access to all data

                current_synthetic = list(state.get("synthetic_placeholders", []) or [])
                current_subprompt_results = state.get("subprompt_results", {}) or {}

                # Add parallel results to synthetic_placeholders if not already there
                existing_fields = {p.get("field") for p in current_synthetic if isinstance(p, dict)}
                for name, content in current_subprompt_results.items():
                    if name not in existing_fields:
                        current_synthetic.append({"field": name, "value": content})

                # Clear sub-prompt execution state but KEEP the accumulated results
                return Command(
                    goto="provide_placeholders",
                    update={
                        "notes": {"type": "override", "value": []},
                        "messages": [final_report],
                        "final_report": final_report.content,
                        "step_index": step_index + 1,
                        "sequential_context": {"type": "override", "value": []},
                        # CRITICAL: Keep subprompt_results for assembly step placeholder substitution
                        # Don't clear them! Assembly step needs [Step_4__Firmographics] etc.
                        "sub_prompts_phase": "announce",
                        "sequential_branches": [],
                        "parallel_branches": [],
                        "synthetic_placeholders": current_synthetic,
                    },
                )

            # Final step complete
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                "notes": {"type": "override", "value": []},
                "sequential_context": {"type": "override", "value": []},
                "subprompt_results": {"type": "override", "value": {}},
            }

        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1

                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded and model limit unknown. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            "notes": {"type": "override", "value": []},
                            "sequential_context": {"type": "override", "value": []},
                            "subprompt_results": {"type": "override", "value": {}},
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
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    "notes": {"type": "override", "value": []},
                    "sequential_context": {"type": "override", "value": []},
                    "subprompt_results": {"type": "override", "value": {}},
                }

    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        "notes": {"type": "override", "value": []},
        "sequential_context": {"type": "override", "value": []},
        "subprompt_results": {"type": "override", "value": {}},
    }


# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)

deep_researcher_builder.add_node("provide_placeholders", provide_placeholders)
deep_researcher_builder.add_node("prepare_step", prepare_step)
deep_researcher_builder.add_node("run_sub_prompts", run_sub_prompts)
deep_researcher_builder.add_node("parallel_subprompt", parallel_subprompt)
deep_researcher_builder.add_node("fan_out_parallel", fan_out_parallel)
deep_researcher_builder.add_node("collect_parallel", collect_parallel)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)  # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)  # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)  # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

deep_researcher_builder.add_edge(START, "provide_placeholders")
deep_researcher_builder.add_conditional_edges(
    "fan_out_parallel",
    route_parallel_execution,
)
deep_researcher_builder.add_edge("fan_out_parallel", "collect_parallel")
deep_researcher_builder.add_edge("parallel_subprompt", "collect_parallel")
deep_researcher_builder.add_edge("clarify_with_user", "write_research_brief")
deep_researcher_builder.add_edge("write_research_brief", "research_supervisor")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
