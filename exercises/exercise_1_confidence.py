"""Exercise 1 — Confidence scoring + routing.

Build a small LangGraph that fetches a PR, analyzes it, then routes to one of
three terminal nodes by confidence. Goal: see the three branches print
different messages on different PRs.

"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title, "pr_diff": pr.diff,
        "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha,
    }


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    # Gemini supports function calling through its OpenAI-compatible endpoint.
    # LangChain validates the tool arguments into the requested Pydantic object.
    llm = get_llm().with_structured_output(PRAnalysis, method="function_calling")
    system_prompt = """You are a senior software engineer reviewing a pull request.
Analyze the supplied diff carefully. Return a complete, actionable code review:
summarize the change, identify concrete risks, and propose comments with file and
line numbers when possible. Report calibrated confidence in the completeness and
correctness of your review, not confidence that the code itself is safe.

Use high confidence only for a small, unambiguous change with no material open
questions, no substantive review comments, and no human decision needed. Use
medium confidence when a human should check an edge case or design decision.
A small schema change with a migration question must be medium confidence
(0.58 through 0.72), never auto-approve. Use confidence below 0.58 for changes involving security,
authentication, sensitive data, or complex/irreversible migrations where
specific human context is needed; include focused escalation_questions in that
case.

Security calibration is mandatory: if a diff includes authentication, password
hashing, token or secret storage, cloud synchronization of credentials, SQL
injection risk, or another security-sensitive change, set confidence to 0.55 or
lower. These changes always need targeted reviewer questions and must route to
escalation, even when the vulnerabilities are obvious.

Return JSON with exactly these fields:
- summary: string
- risk_factors: array of strings
- comments: array of objects with file (string), line (integer or null),
  severity (one of nit, suggestion, issue, blocker), and body (string)
- confidence: number from 0 to 1
- confidence_reasoning: string
- escalation_questions: array of strings
Do not use alternate field names such as risk_assessment or comment."""
    user_prompt = (
        f"Pull request title: {state['pr_title']}\n\n"
        f"Unified diff:\n{state['pr_diff']}"
    )
    with console.status("[dim]LLM thinking...[/dim]"):
        analysis = llm.invoke([("system", system_prompt), ("user", user_prompt)])
    return {"analysis": analysis}



def node_route(state: ReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    confidence = state["analysis"].confidence
    if confidence >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif confidence < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"

    console.print(f"  confidence = {confidence:.0%} -> {decision}")
    return {"decision": decision}



def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]✓ AUTO APPROVE[/green] — high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]✓ HUMAN APPROVAL[/yellow] — placeholder, exercise 2 will pause here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]✓ ESCALATE[/red] — placeholder, exercise 3 will ask the reviewer questions")
    return {"final_action": "pending_escalation"}


def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("fetch_pr", node_fetch_pr)
    g.add_node("analyze", node_analyze)
    g.add_node("route", node_route)
    g.add_node("auto_approve", node_auto_approve)
    g.add_node("human_approval", node_human_approval)
    g.add_node("escalate", node_escalate)

    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda state: state["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", END)
    g.add_edge("escalate", END)
    return g.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 — confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    final = app.invoke({"pr_url": args.pr})

    console.rule("Final")
    console.print(f"confidence = {final['analysis'].confidence:.0%}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()
