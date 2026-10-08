"""
Streamlit Web Dashboard for 'Why The Code Is Like This'.
Provides an interactive chat interface, knowledge graph neighborhood explorer, and decision timeline.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from settings import get_settings
from src.graph.embedding_generator import EmbeddingGenerator
from src.graph.neo4j_manager import Neo4jManager
from src.models import Confidence
from src.query.answer_engine import AnswerEngine
from src.query.retriever import WhyRetriever

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

        try:
            stats = neo4j_mgr.get_stats()
            st.sidebar.markdown("### 📊 Knowledge Graph Stats")
            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                st.metric("Decisions", stats.total_decisions or stats.nodes)
                st.metric("Active", stats.active_decisions)
                st.metric("Commits", stats.total_commits)
                st.metric("People", stats.total_people)
            with col_b:
                st.metric("Superseded", stats.superseded_decisions)
                st.metric("Pull Requests", stats.total_prs)
                st.metric("Issues", stats.total_issues)
                st.metric("Code Entities", stats.total_code_entities)
        except Exception as e:
            logger.debug(f"Sidebar stats loading note: {e}")

    # Initialize Retrieval & Answer Engine
    if neo4j_mgr:
        embedding_gen = EmbeddingGenerator(settings.google_api_key, settings.embedding_model)
        retriever = WhyRetriever(neo4j_mgr, embedding_gen)
        answer_engine = AnswerEngine(settings.google_api_key, settings.gemini_model, retriever)
    else:
        answer_engine = None

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
                    answer = answer_engine.answer(question_input.strip())

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

                # Evidence Cards
                if answer.evidence:
                    st.markdown("### 📎 Evidence Chain")
                    for i, ev in enumerate(answer.evidence, 1):
                        with st.expander(f"{i}. [{ev.source_type.upper()}] {ev.title or ev.source_id}"):
                            st.write(f"**Summary:** {ev.summary}")
                            if ev.person:
                                st.write(f"**Author / Contributor:** `@{ev.person}`")
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
                decisions_records = neo4j_mgr.execute_query(
                    "MATCH (d:Decision) RETURN d.id AS id, d.title AS title LIMIT 50"
                )
                if decisions_records:
                    dec_options = {
                        f"{r.get('title', 'Decision')} ({r.get('id', '')})": r.get("id")
                        for r in decisions_records
                        if r.get("id")
                    }
                    selected_dec_label = st.selectbox("Select a Decision node to inspect:", list(dec_options.keys()))
                    selected_id = dec_options[selected_dec_label]

                    context_data = neo4j_mgr.get_decision_with_context(selected_id)
                    dec_node = context_data.get("decision") or {}

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown("#### Decision Details")
                        st.write(f"**Title:** {dec_node.get('title', 'N/A')}")
                        st.write(f"**Type:** {dec_node.get('decision_type', dec_node.get('type', 'N/A'))}")
                        st.write(f"**Status:** {dec_node.get('status', 'ACTIVE')}")
                        st.write(f"**Rationale:** {dec_node.get('rationale', 'N/A')}")
                        if context_data.get("author"):
                            st.write(f"**Decided By:** @{context_data['author'].get('login', 'N/A')}")

                    with col2:
                        st.markdown("#### Connected Subgraph")
                        if HAS_AGRAPH:
                            nodes = []
                            edges = []
                            added_ids = set()

                            # Root Decision Node
                            root_id = str(selected_id)
                            nodes.append(
                                Node(
                                    id=root_id,
                                    label=dec_node.get("title", root_id)[:25] + "...",
                                    color="#1E88E5",
                                    size=25,
                                )
                            )
                            added_ids.add(root_id)

                            # Connected PR
                            if context_data.get("pr"):
                                pr_id = f"PR-{context_data['pr'].get('number')}"
                                nodes.append(Node(id=pr_id, label=f"PR #{context_data['pr'].get('number')}", color="#43A047", size=18))
                                edges.append(Edge(source=root_id, target=pr_id, label="DISCUSSED_IN"))

                            # Connected Commit
                            if context_data.get("commit"):
                                c_id = f"C-{context_data['commit'].get('sha', '')[:7]}"
                                nodes.append(Node(id=c_id, label=c_id, color="#FB8C00", size=18))
                                edges.append(Edge(source=root_id, target=c_id, label="IMPLEMENTED_IN"))

                            # Connected Entities
                            for ent in context_data.get("entities", [])[:5]:
                                ent_name = ent.get("name") or ent.get("path", "file")
                                ent_id = f"ENT-{ent_name}"
                                if ent_id not in added_ids:
                                    nodes.append(Node(id=ent_id, label=ent_name[:20], color="#8E24AA", size=15))
                                    edges.append(Edge(source=root_id, target=ent_id, label="AFFECTS"))
                                    added_ids.add(ent_id)

                            # Superseded Lineage
                            for sup in context_data.get("superseded", []):
                                s_id = f"SUP-{sup.get('id', '')}"
                                if s_id not in added_ids:
                                    nodes.append(Node(id=s_id, label=f"Old: {sup.get('title', '')[:15]}", color="#E53935", size=15))
                                    edges.append(Edge(source=root_id, target=s_id, label="SUPERSEDES"))
                                    added_ids.add(s_id)

                            config = Config(width=650, height=450, directed=True, nodeHighlightBehavior=True)
                            agraph(nodes=nodes, edges=edges, config=config)
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
                timeline_query = """
                MATCH (d:Decision)
                OPTIONAL MATCH (newer:Decision)-[:SUPERSEDES]->(d)
                RETURN d, newer
                ORDER BY d.id DESC
                LIMIT 50
                """
                records = neo4j_mgr.execute_query(timeline_query)
                if records:
                    for rec in records:
                        d = rec.get("d") or {}
                        newer = rec.get("newer")
                        title = d.get("title", "Untitled Decision")
                        dec_type = d.get("decision_type", d.get("type", "ARCHITECTURE"))
                        status = d.get("status", "ACTIVE")
                        rationale = d.get("rationale", "No rationale documented.")
                        problem = d.get("problem_context", "")

                        is_superseded = status == "SUPERSEDED" or bool(newer)
                        badge = "⚠️ SUPERSEDED" if is_superseded else "✅ ACTIVE"
                        expander_title = f"{title} [{dec_type}] — {badge}"

                        with st.expander(expander_title):
                            st.write(f"**Decision ID:** `{d.get('id', 'N/A')}`")
                            if problem:
                                st.write(f"**Problem Context:** {problem}")
                            st.write(f"**Rationale:** {rationale}")
                            if is_superseded and newer:
                                st.warning(f"This decision was superseded by: **{newer.get('title', newer.get('id', ''))}**")
                else:
                    st.info("No decisions recorded in knowledge graph yet. Run ingestion to view timeline.")
            except Exception as e:
                st.error(f"Error loading timeline: {e}")


if __name__ == "__main__":
    main()
