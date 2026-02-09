"""
LangGraph workflow orchestration module.

This module defines the state machine that coordinates the supervisor and
coding agents. It uses LangGraph to create a directed graph where each node
represents a step in the question-answering process.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.agents import CodingAgent, SupervisorAgent, build_coding_config, build_supervisor_config
from src.db_manager import DuckDBManager
from src.schema_manager import SchemaManager


class GraphState(TypedDict):
    """
    State schema for the LangGraph workflow.
    
    This TypedDict defines all the data that flows through the graph.
    Each node can read and modify the state as needed.
    
    Attributes:
        messages: Conversation history (uses LangGraph's add_messages reducer)
        question: Current question being processed (may be sub-question)
        original_question: Original user question (preserved for aggregation)
        schema_info: Schema description for agent context
        query: Generated SQL query (set by coding node)
        results: Query execution results (set by execute node)
        answer: Final synthesized answer (set by synthesize node)
        error: Error message if something goes wrong
        needs_query: Boolean flag indicating if a query is needed
        query_is_valid: Boolean indicating if the generated query passed verification
        query_feedback: Feedback string if the query was invalid
        retries: Number of correction attempts
        is_complex: Boolean indicating if question was decomposed
        sub_questions: List of decomposed sub-questions
        current_sub_question_index: Current position in sub-question queue
        sub_results: Accumulated results from sub-question processing
    """
    # Conversation history - uses add_messages reducer for proper message merging
    messages: Annotated[List[BaseMessage], add_messages]
    
    question: str
    original_question: Optional[str]
    schema_info: str
    query: Optional[str]
    results: Optional[list[Dict[str, Any]]]
    answer: Optional[str]
    error: Optional[str]
    needs_query: Optional[bool]
    query_is_valid: Optional[bool]
    query_feedback: Optional[str]
    retries: Optional[int]
    is_complex: Optional[bool]
    sub_questions: Optional[list[str]]
    current_sub_question_index: Optional[int]
    sub_results: Optional[list[Dict[str, Any]]]


class AgenticGraph:
    """
    LangGraph workflow orchestrating supervisor and coding agents.
    
    This class builds and manages a state machine with the following flow:
    1. Supervisor analyzes question → determines if query needed
    2. If query needed → Coding agent generates SQL → Execute → Synthesize answer
    3. If no query needed → Supervisor provides direct answer
    
    The graph uses conditional edges to route based on the supervisor's decision.
    """

    def __init__(self, schema_manager: SchemaManager, db: DuckDBManager, checkpointer=None) -> None:
        """
        Initialize the graph with dependencies.
        
        Args:
            schema_manager: SchemaManager instance for schema information
            db: DuckDBManager instance for query execution
            checkpointer: Optional LangGraph checkpointer for memory persistence.
                         If provided, enables conversation history across invocations.
        """
        self.schema_manager = schema_manager
        self.db = db
        self.max_retries = 3
        self.checkpointer = checkpointer
        
        # Initialize agents with their configurations
        supervisor_cfg = build_supervisor_config()
        coding_cfg = build_coding_config()
        self.supervisor = SupervisorAgent(supervisor_cfg)
        self.coder = CodingAgent(coding_cfg)
        
        # Build and compile the LangGraph workflow
        self.graph = self._build_graph()

    def _supervisor_node(self, state: GraphState) -> GraphState:
        """
        Supervisor node: Analyzes the question and determines if query is needed.
        
        This is the entry point of the graph. The supervisor agent analyzes
        the user's question and decides whether a database query is required
        to answer it.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with needs_query flag set and potentially an answer
        """
        try:
            # Ask supervisor to analyze the question
            analysis = self.supervisor.analyze_question(
                question=state["question"],
                schema_info=state["schema_info"]
            )
            
            # Check if supervisor determined a query is needed
            needs_query = "NEED_QUERY" in analysis.upper()
            
            # If no query needed, supervisor can provide direct answer
            if not needs_query:
                state["answer"] = analysis
                state["needs_query"] = False
            else:
                state["needs_query"] = True
                state["retries"] = 0  # Initialize retry counter
            
            return state
            
        except Exception as e:
            # Handle errors in supervisor analysis
            state["error"] = f"Supervisor error: {e}"
            state["needs_query"] = False
            return state

    def _coding_node(self, state: GraphState) -> GraphState:
        """
        Coding node: Generates SQL query from the question.
        
        This node uses the coding agent to translate the natural language
        question into a SQL query that can be executed against DuckDB.
        
        Args:
            state: Current graph state (must have question and schema_info)
            
        Returns:
            Updated state with generated SQL query
        """
        try:
            feedback = state.get("query_feedback")
            if feedback:
                print(f"--- RETRYING QUERY GENERATION (Feedback: {feedback}) ---")
            else:
                print("--- GENERATING INITIAL QUERY ---")

            # Ask coding agent to generate SQL query, passing feedback and previous query if any
            query = self.coder.generate_query(
                question=state["question"],
                schema_info=state["schema_info"],
                feedback=feedback,
                previous_query=state.get("query") if feedback else None
            )
            
            # Store the generated query in state
            state["query"] = query
            return state
            
        except Exception as e:
            # Handle errors in query generation - increment retries to prevent infinite loop
            state["error"] = f"Query generation error: {e}"
            current_retries = state.get("retries", 0)
            state["retries"] = current_retries + 1
            print(f"⚠️ Query generation error (attempt {state['retries']}/{self.max_retries + 1}): {e}")
            return state

    def _execute_node(self, state: GraphState) -> GraphState:
        """
        Execute node: Runs the SQL query in DuckDB.
        
        This node takes the generated SQL query and executes it against
        the DuckDB database, storing the results in the state.
        
        Args:
            state: Current graph state (must have query)
            
        Returns:
            Updated state with query results
        """
        # Skip execution if there's already an error from previous nodes
        if state.get("error"):
            return state
        
        try:
            # Validate that a query exists
            if not state.get("query"):
                state["error"] = "No query to execute"
                return state
            
            # Execute the query and store results
            results = self.db.query(state["query"])
            state["results"] = results
            return state
            
        except Exception as e:
            # Handle query execution errors
            state["error"] = f"Query execution error: {e}"
            return state

    def _synthesize_node(self, state: GraphState) -> GraphState:
        """
        Synthesize node: Creates final natural language answer from results.
        
        This node uses the supervisor agent to synthesize a clear, natural
        language answer from the query results. It combines the original
        question, the SQL query, and the results into a coherent response.
        
        Args:
            state: Current graph state (must have question, query, and results)
            
        Returns:
            Updated state with final answer
        """
        try:
            # If there's an error, include it in the answer
            if state.get("error"):
                state["answer"] = f"I encountered an error: {state['error']}"
                return state
            
            # Ask supervisor to synthesize answer from results
            answer = self.supervisor.synthesize_answer(
                question=state["question"],
                query=state.get("query", ""),
                results=state.get("results", []),
                schema_info=state["schema_info"]
            )
            
            # Store the synthesized answer
            state["answer"] = answer
            return state
            
        except Exception as e:
            # Handle synthesis errors
            state["error"] = f"Synthesis error: {e}"
            state["answer"] = f"I encountered an error while synthesizing the answer: {e}"
            return state

    def _should_query(self, state: GraphState) -> Literal["query", "end"]:
        """
        Conditional edge function: Determines routing after supervisor node.
        
        This function is used by LangGraph to decide which path to take
        after the supervisor analyzes the question.
        
        Args:
            state: Current graph state
            
        Returns:
            "query" if a query is needed, "end" if supervisor provided direct answer
        """
        if state.get("needs_query"):
            return "query"
        return "end"

    def _verify_node(self, state: GraphState) -> GraphState:
        """
        Verification node: Checks the generated SQL against the actual database schema.
        
        This uses DuckDB's EXPLAIN command to deterministically catch Binder Errors
        (hallucinated columns/tables) before execution.
        """
        # Handle error from coding node - still need to increment retries
        if state.get("error"):
            current_retries = state.get("retries", 0)
            # Only increment if we haven't already in the coding node
            if current_retries <= self.max_retries:
                print(f"⚠️ Skipping verification due to error: {state['error']}")
            state["query_is_valid"] = False
            state["query_feedback"] = state.get("error", "Unknown error")
            # Clear the error so we can retry
            state["error"] = None
            return state
            
        try:
            print(f"--- VERIFYING QUERY (Attempt {state.get('retries', 0) + 1}/{self.max_retries + 1}) ---")
            # Deterministic check using DuckDB engine
            verification = self.db.validate_schema(state["query"])
            
            state["query_is_valid"] = verification.get("valid", True)
            state["query_feedback"] = verification.get("error", "")
            
            if not state["query_is_valid"]:
                print(f"❌ Verification FAILED: {state['query_feedback']}")
                # Increment retries if invalid
                current_retries = state.get("retries", 0)
                state["retries"] = current_retries + 1
            else:
                print("✅ Verification PASSED")
                
            return state
            
        except Exception as e:
            # If verification fails unexpectedly, we log it
            print(f"⚠️ Verification System Error: {e}")
            state["query_is_valid"] = True # Fail open to avoid deadlock
            state["query_feedback"] = f"Verification system error: {e}"
            return state

    def _check_verification(self, state: GraphState) -> Literal["execute", "retry_coding"]:
        """
        Route after verification.
        If valid -> execute.
        If invalid -> retry coding (coding node handles feedback).
        """
        if state.get("query_is_valid"):
            return "execute"
        
        # Check max retries
        if state.get("retries", 0) >= self.max_retries:
            # Give up and try to run it anyway, or fail gracefully
            # Let's run it and let the execute node handle the error if it fails
            return "execute"
            
        return "retry_coding"

    # ==================== QUERY EXPANSION NODES ====================

    def _expansion_node(self, state: GraphState) -> GraphState:
        """
        Expansion node: Analyzes question complexity and decomposes if needed.
        
        This node uses the supervisor agent to determine if the question
        should be broken down into simpler sub-questions.
        """
        try:
            print("--- ANALYZING QUESTION COMPLEXITY ---")
            result = self.supervisor.detect_complexity(
                question=state["question"],
                schema_info=state["schema_info"]
            )
            
            state["is_complex"] = result.get("is_complex", False)
            state["original_question"] = result.get("original_question", state["question"])
            
            if state["is_complex"]:
                state["sub_questions"] = result.get("sub_questions", [])
                state["current_sub_question_index"] = 0
                state["sub_results"] = []
                print(f"✂️ Question decomposed into {len(state['sub_questions'])} sub-questions:")
                for i, sq in enumerate(state["sub_questions"], 1):
                    print(f"   {i}. {sq}")
            else:
                print("📝 Simple question - no decomposition needed")
                reasoning = result.get("reasoning", "")
                if reasoning:
                    print(f"   Reason: {reasoning}")
            
            return state
            
        except Exception as e:
            print(f"⚠️ Expansion error: {e}")
            state["is_complex"] = False
            return state

    def _sub_question_node(self, state: GraphState) -> GraphState:
        """
        Sub-question node: Prepares the current sub-question for processing.
        
        This node sets up the state for the coding agent to process
        the current sub-question from the queue.
        """
        idx = state.get("current_sub_question_index", 0)
        sub_questions = state.get("sub_questions", [])
        
        if idx < len(sub_questions):
            current_sub_q = sub_questions[idx]
            print(f"\n--- PROCESSING SUB-QUESTION {idx + 1}/{len(sub_questions)} ---")
            print(f"   \"{current_sub_q}\"")
            
            # Override question with current sub-question
            state["question"] = current_sub_q
            # Reset ALL query-related state for new generation (prevent leakage)
            state["query"] = None
            state["results"] = None
            state["query_is_valid"] = None
            state["query_feedback"] = None
            state["error"] = None  # Clear any error from previous sub-question
            state["retries"] = 0
        
        return state

    def _collect_result_node(self, state: GraphState) -> GraphState:
        """
        Collect result node: Stores the current sub-question result and advances queue.
        
        This node saves the query and results for the current sub-question,
        then increments the index to move to the next sub-question.
        """
        idx = state.get("current_sub_question_index", 0)
        sub_questions = state.get("sub_questions", [])
        
        if idx < len(sub_questions):
            sub_result = {
                "sub_question": sub_questions[idx],
                "query": state.get("query"),
                "results": state.get("results", []),
            }
            
            if state.get("sub_results") is None:
                state["sub_results"] = []
            state["sub_results"].append(sub_result)
            
            print(f"✅ Collected result for sub-question {idx + 1}")
            
            # Advance to next sub-question
            state["current_sub_question_index"] = idx + 1
        
        return state

    def _aggregation_node(self, state: GraphState) -> GraphState:
        """
        Aggregation node: Combines all sub-question results into final answer.
        
        This node uses the supervisor agent to synthesize a comprehensive
        answer from all the partial results.
        """
        try:
            print("\n--- AGGREGATING SUB-QUESTION RESULTS ---")
            
            sub_results = state.get("sub_results", [])
            if not sub_results:
                state["answer"] = "No results to aggregate."
                return state
            
            print(f"   Combining {len(sub_results)} partial results...")
            
            answer = self.supervisor.aggregate_results(
                original_question=state.get("original_question", state["question"]),
                sub_results=sub_results,
                schema_info=state["schema_info"]
            )
            
            state["answer"] = answer
            return state
            
        except Exception as e:
            state["error"] = f"Aggregation error: {e}"
            state["answer"] = f"I encountered an error while aggregating results: {e}"
            return state

    def _check_complexity(self, state: GraphState) -> Literal["expand", "simple"]:
        """
        Route after expansion node based on question complexity.
        """
        if state.get("is_complex") and state.get("sub_questions"):
            return "expand"
        return "simple"

    def _has_more_sub_questions(self, state: GraphState) -> Literal["more", "aggregate"]:
        """
        Route after collecting a sub-question result.
        Check if there are more sub-questions to process.
        """
        idx = state.get("current_sub_question_index", 0)
        sub_questions = state.get("sub_questions", [])
        
        if idx < len(sub_questions):
            return "more"
        return "aggregate"


    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow with nodes and edges.
        
        This method constructs the state machine with query expansion support:
        - supervisor → expansion → [complex?] → expand path or simple path
        - expand path: sub_question → coding → verify → execute → collect → [more?] → loop or aggregate
        - simple path: coding → verify → execute → synthesize
        
        Returns:
            Compiled LangGraph workflow ready to execute
        """
        # Create a new StateGraph with our state schema
        workflow = StateGraph(GraphState)
        
        # Add all nodes to the graph
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("expansion", self._expansion_node)
        workflow.add_node("sub_question", self._sub_question_node)
        workflow.add_node("coding", self._coding_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("collect", self._collect_result_node)
        workflow.add_node("aggregation", self._aggregation_node)
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Set the entry point (where the graph starts)
        workflow.set_entry_point("supervisor")
        
        # Supervisor → [needs_query?] → expansion OR end
        workflow.add_conditional_edges(
            "supervisor",
            self._should_query,
            {
                "query": "expansion",
                "end": END,
            }
        )
        
        # Expansion → [complex?] → sub_question OR coding (simple path)
        workflow.add_conditional_edges(
            "expansion",
            self._check_complexity,
            {
                "expand": "sub_question",
                "simple": "coding",
            }
        )
        
        # Sub-question preparation → Coding
        workflow.add_edge("sub_question", "coding")
        
        # Coding → Verify
        workflow.add_edge("coding", "verify")
        
        # Verify → [valid?] → Execute OR Retry Coding
        workflow.add_conditional_edges(
            "verify",
            self._check_verification,
            {
                "execute": "execute",
                "retry_coding": "coding",
            }
        )
        
        # Execute → [is_complex?] → collect (for aggregation) OR synthesize (simple path)
        workflow.add_conditional_edges(
            "execute",
            lambda state: "collect" if state.get("is_complex") else "synthesize",
            {
                "collect": "collect",
                "synthesize": "synthesize",
            }
        )
        
        # Collect → [more sub-questions?] → sub_question OR aggregation
        workflow.add_conditional_edges(
            "collect",
            self._has_more_sub_questions,
            {
                "more": "sub_question",
                "aggregate": "aggregation",
            }
        )
        
        # Aggregation → End
        workflow.add_edge("aggregation", END)
        
        # Synthesize → End (simple path)
        workflow.add_edge("synthesize", END)
        
        # Compile and return the graph (with checkpointer if provided)
        return workflow.compile(checkpointer=self.checkpointer)

    def run(self, question: str, config: Optional[Dict[str, Any]] = None) -> GraphState:
        """
        Run the graph with a user question.
        
        This is the main entry point for processing questions. It:
        1. Gets schema information from the schema manager
        2. Creates initial state with the question
        3. Invokes the graph workflow
        4. Returns the final state with the answer
        
        Args:
            question: User's natural language question
            config: Optional config dict with 'configurable' containing 'thread_id'
                   for memory persistence. Use MemoryManager.get_config() to create.
            
        Returns:
            Final graph state containing the answer (or error)
        """
        # Get schema summary for agent context
        schema_summary = self.schema_manager.describe()
        
        # Create initial state with user message
        initial_state: GraphState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "original_question": question,  # Preserved for aggregation
            "schema_info": schema_summary,
            "query": None,
            "results": None,
            "answer": None,
            "error": None,
            "needs_query": None,
            "query_is_valid": None,
            "query_feedback": None,
            "retries": 0,
            # Query expansion fields
            "is_complex": None,
            "sub_questions": None,
            "current_sub_question_index": 0,
            "sub_results": None,
        }
        
        # Run the graph workflow with optional config for checkpointing
        # LangGraph will execute nodes in order based on edges
        if config is not None:
            final_state = self.graph.invoke(initial_state, config)
        else:
            final_state = self.graph.invoke(initial_state)
        
        # Add assistant message with the answer to state for memory
        if final_state.get("answer"):
            final_state["messages"] = final_state.get("messages", []) + [
                AIMessage(content=final_state["answer"])
            ]
        
        return final_state
