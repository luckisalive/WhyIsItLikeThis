"""
Streamlit Web Dashboard for 'Why The Code Is Like This'.
Provides an interactive chat interface, knowledge graph neighborhood explorer, and decision timeline.
Framework-agnostic services handle data operations; this module handles rendering.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from settings import get_settings
from src.graph.neo4j_manager import Neo4jManager
from src.models import Confidence
from src.services import ask, decisions, graph, stats

try:
    from streamlit_agraph import Config, Edge, Node, agraph
    HAS_AGRAPH = True
except ImportError:
    HAS_AGRAPH = False

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Why The Code Is Like This",
    layout="wide",
    page_icon="🔍",
)


@st.cache_resource
def get_app_settings():
    return get_settings()


def get_neo4j_manager(settings):
    try:
        mgr = Neo4jManager(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
        mgr.connect()
        return mgr
    except Exception as e:
        logger.warning(f"Could not connect to Neo4j: {e}")
        return None


def main():
    settings = get_app_settings()

    # Sidebar: Repository Context & Connection Status
    st.sidebar.title("🔍 Why The Code Is Like This")
    st.sidebar.markdown(f"**Repository:** `{settings.target_repo}`")

    neo4j_mgr = get_neo4j_manager(settings)

    if not neo4j_mgr:
        st.sidebar.error("⚠️ Neo4j Disconnected")
        st.sidebar.info(
            "Please check your `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` in `.env`. "
            "If using AuraDB Free, ensure your instance is resumed in the Neo4j Console."
        )
    else:
        st.sidebar.success("🟢 Neo4j Connected")
        if st.sidebar.button("🔄 Refresh Statistics"):
            st.rerun()

        stats_result = stats.get_stats(neo4j_mgr)
        if stats_result.success and stats_result.stats:
            stats_data = stats_result.stats
            st.sidebar.markdown("### 📊 Knowledge Graph Stats")
            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                st.metric("Decisions", stats_data.total_decisions or stats_data.nodes)
                st.metric("Active", stats_data.active_decisions)
                st.metric("Commits", stats_data.total_commits)
                st.metric("People", stats_data.total_people)
            with col_b:
                st.metric("Superseded", stats_data.superseded_decisions)
                st.metric("Pull Requests", stats_data.total_prs)
                st.metric("Issues", stats_data.total_issues)
                st.metric("Code Entities", stats_data.total_code_entities)
        elif stats_result.error:
            logger.debug(f"Sidebar stats loading note: {stats_result.error.message}")

    # Initialize Answer Engine via service
    answer_engine = ask.create_answer_engine(neo4j_mgr, settings) if neo4j_mgr else None

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Ask Why", "🕸️ Knowledge Graph Explorer", "📅 Decision Timeline"])

    # ─────────────────────────────────────────────────────────
    # TAB 1: Chat / Question Answering
    # ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Ask Why the Code is the Way it is")
        st.markdown(
            "Query the repository history for the **rationale** behind architecture, configuration, "
            "and design choices. Every answer includes verifiable evidence from commits, PRs, and discussions."
        )

        example_questions = [
            "Why does FastAPI use Starlette as its foundation?",
            "Why was Pydantic chosen for request and response validation?",
            "Why is the retry or timeout configuration set to its current value?",
            "Who decided to support both async and def route handlers?",
        ]
        selected_example = st.selectbox("💡 Or choose an example question:", ["(Custom question)"] + example_questions)

        question_input = st.text_input(
            "Enter your question:",
            value="" if selected_example == "(Custom question)" else selected_example,
            placeholder="e.g. Why was dependency injection implemented this way?",
        )

        if st.button("🔍 Answer with Evidence Chain", type="primary") or (question_input and selected_example != "(Custom question)"):
            if not question_input.strip():
                st.warning("Please enter a question.")
            elif not answer_engine:
                st.error("Neo4j database connection required to search knowledge graph.")
            else:
                with st.spinner("Traversing knowledge graph & analyzing decision provenance..."):
                    ask_result = ask.answer(question_input.strip(), neo4j_mgr, settings, answer_engine)

                if not ask_result.success:
                    st.error(ask_result.error.message if ask_result.error else "Failed to get answer.")
                elif ask_result.answer:
                    answer = ask_result.answer
                    # Display Confidence Banner
                    if answer.confidence == Confidence.HIGH:
                        st.success(f"🟢 **Confidence: HIGH** — {answer.confidence_explanation}")
                    elif answer.confidence == Confidence.MEDIUM:
                        st.warning(f"🟡 **Confidence: MEDIUM** — {answer.confidence_explanation}")
                    else:
                        st.error(f"🔴 **Confidence: LOW** — {answer.confidence_explanation}")

                    # Supersession Banner
                    if answer.supersession:
                        st.warning(
                            f"⚠️ **Supersession Alert**: '{answer.supersession.superseded_decision}' "
                            f"was later overridden by **{answer.supersession.superseded_by}**.\n\n"
                            f"*Reason:* {answer.supersession.reason}"
                        )

                    # Primary Answer Box
                    st.markdown("### 💡 Architectural Explanation")
                    st.markdown(answer.answer_text)

                    # Point of Contact
                    if answer.ask_person:
                        st.info(f"👤 **Person to consult for more context:** `@{answer.ask_person}`")

                    # Related Decisions
                    if answer.related_decisions:
                        st.caption(f"🔗 **Related Decisions:** {', '.join(answer.related_decisions)}")

                    # Evidence Cards
                    if answer.evidence:
                        st.markdown("### 📎 Evidence Chain")
                        for i, ev in enumerate(answer.evidence, 1):
                            with st.expander(f"{i}. [{ev.source_type.upper()}] {ev.title or ev.source_id}"):
                                st.write(f"**Summary:** {ev.summary}")
                                if ev.person:
                                    st.write(f"**Author / Contributor:** `@{ev.person}`")
                                if ev.date:
                                    st.write(f"**Date:** `{ev.date}`")
                                if ev.url:
                                    st.markdown(f"**Direct Link:** [{ev.url}]({ev.url})")

    # ─────────────────────────────────────────────────────────
    # TAB 2: Knowledge Graph Visualization
    # ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Interactive Knowledge Graph Explorer")
        st.markdown("Explore how decisions connect to pull requests, commits, issues, people, and code entities.")

        if not neo4j_mgr:
            st.info("Connect to Neo4j AuraDB to visualize graph relationships.")
        else:
            try:
                decisions_result = decisions.list_decisions(neo4j_mgr, limit=50)
                if not decisions_result.success:
                    st.error(f"Error loading graph explorer: {decisions_result.error.message if decisions_result.error else 'Unknown error'}")
                elif decisions_result.items:
                    dec_options = {
                        f"{item.get('title', 'Decision')} ({item.get('id', '')})": item.get("id")
                        for item in decisions_result.items
                        if item.get("id")
                    }
                    selected_dec_label = st.selectbox("Select a Decision node to inspect:", list(dec_options.keys()))
                    selected_id = dec_options[selected_dec_label]

                    graph_result = graph.get_decision_graph(neo4j_mgr, selected_id)
                    if not graph_result.success:
                        st.error(f"Error loading graph explorer: {graph_result.error.message if graph_result.error else 'Unknown error'}")
                    else:
                        dec_node = graph_result.decision or {}

                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown("#### Decision Details")
                            st.write(f"**Title:** {dec_node.get('title', 'N/A')}")
                            st.write(f"**Type:** {dec_node.get('decision_type', dec_node.get('type', 'N/A'))}")
                            st.write(f"**Status:** {dec_node.get('status', 'ACTIVE')}")
                            st.write(f"**Rationale:** {dec_node.get('rationale', 'N/A')}")
                            if graph_result.author:
                                st.write(f"**Decided By:** @{graph_result.author.get('login', 'N/A')}")

                        with col2:
                            st.markdown("#### Connected Subgraph")
                            if HAS_AGRAPH:
                                nodes = []
                                edges = []
                                root_node_id = f"DEC-{selected_id}"

                                for gn in graph_result.nodes:
                                    if gn.type == "Decision" and (gn.id == root_node_id or gn.id == str(selected_id)):
                                        nodes.append(Node(id=gn.id, label=gn.label[:25] + "...", color="#1E88E5", size=25))
                                    elif gn.type == "PullRequest":
                                        nodes.append(Node(id=gn.id, label=gn.label, color="#43A047", size=18))
                                    elif gn.type == "Commit":
                                        nodes.append(Node(id=gn.id, label=gn.label[:20], color="#FB8C00", size=18))
                                    elif gn.type == "Issue":
                                        nodes.append(Node(id=gn.id, label=gn.label[:20], color="#0288D1", size=18))
                                    elif gn.type == "Person":
                                        nodes.append(Node(id=gn.id, label=gn.label, color="#7B1FA2", size=18))
                                    elif gn.type == "CodeEntity":
                                        nodes.append(Node(id=gn.id, label=gn.label[:20], color="#8E24AA", size=15))
                                    elif gn.type == "Decision":
                                        # Older or Newer superseding decision
                                        is_newer = gn.label.startswith("New:") or gn.label.startswith("Newer:")
                                        color = "#D32F2F" if is_newer else "#E53935"
                                        nodes.append(Node(id=gn.id, label=gn.label[:20], color=color, size=18 if is_newer else 15))

                                for ge in graph_result.edges:
                                    edges.append(Edge(source=ge.source, target=ge.target, label=ge.label))

                                config = Config(width=650, height=450, directed=True, nodeHighlightBehavior=True)
                                agraph(nodes=nodes, edges=edges, config=config)

                                if graph_result.hidden_count > 0:
                                    st.caption(f"ℹ️ +{graph_result.hidden_count} additional code entities omitted from graph view.")
                            else:
                                st.info("Install `streamlit-agraph` for interactive graph rendering.")
                else:
                    st.info("No decisions found in the graph yet. Run `python main.py ingest --quick` to populate.")
            except Exception as e:
                st.error(f"Error loading graph explorer: {e}")

    # ─────────────────────────────────────────────────────────
    # TAB 3: Decision Timeline & Evolution
    # ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Architectural Decision Timeline")
        st.markdown("Chronological provenance of all recorded decisions with supersession indicators.")

        if not neo4j_mgr:
            st.info("Connect to Neo4j AuraDB to inspect the decision timeline.")
        else:
            try:
                timeline_result = decisions.list_decisions(neo4j_mgr, limit=50)
                if not timeline_result.success:
                    if timeline_result.error and timeline_result.error.code == "NO_RESULTS":
                        st.info("No decisions recorded in knowledge graph yet. Run ingestion to view timeline.")
                    else:
                        st.error(f"Error loading timeline: {timeline_result.error.message if timeline_result.error else 'Unknown error'}")
                elif timeline_result.items:
                    for item in timeline_result.items:
                        d = item.get("d") or {}
                        newer = item.get("newer")
                        title = item.get("title") or d.get("title", "Untitled Decision")
                        dec_type = item.get("decision_type", "ARCHITECTURE")
                        is_superseded = item.get("is_superseded", False)
                        rationale = item.get("rationale") or d.get("rationale", "No rationale documented.")
                        problem = item.get("problem_context") or d.get("problem_context", "")

                        badge = "⚠️ SUPERSEDED" if is_superseded else "✅ ACTIVE"
                        expander_title = f"{title} [{dec_type}] — {badge}"

                        with st.expander(expander_title):
                            st.write(f"**Decision ID:** `{item.get('id', 'N/A')}`")
                            if problem:
                                st.write(f"**Problem Context:** {problem}")
                            st.write(f"**Rationale:** {rationale}")
                            if is_superseded and newer:
                                newer_title = newer.get("title", newer.get("id", "")) if isinstance(newer, dict) else str(newer)
                                st.warning(f"This decision was superseded by: **{newer_title}**")
                else:
                    st.info("No decisions recorded in knowledge graph yet. Run ingestion to view timeline.")
            except Exception as e:
                st.error(f"Error loading timeline: {e}")


if __name__ == "__main__":
    main()
