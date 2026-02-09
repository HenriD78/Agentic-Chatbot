"""
Agent definitions for the Agentic RAG system.

This module defines two specialized agents:
1. SupervisorAgent: Analyzes questions and synthesizes answers
2. CodingAgent: Generates SQL queries from natural language questions

Both agents use Ollama LLMs via LangChain's ChatOllama interface.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings


@dataclass
class AgentConfig:
    """
    Configuration dataclass for agent initialization.
    
    Attributes:
        host: Ollama API base URL
        model: Name of the Ollama model to use
        temperature: Sampling temperature for generation (0.0-1.0)
    """
    host: str
    model: str
    temperature: float


def build_supervisor_config() -> AgentConfig:
    """
    Build configuration for the supervisor agent.
    
    Returns:
        AgentConfig with settings for ministral-3:3b model
    """
    return AgentConfig(
        host=settings.ollama_host,
        model=settings.supervisor_model,
        temperature=settings.agent_temperature,
    )


def build_coding_config() -> AgentConfig:
    """
    Build configuration for the coding agent.
    
    Returns:
        AgentConfig with settings for qwen2.5-coder:7b model
    """
    return AgentConfig(
        host=settings.ollama_host,
        model=settings.coder_model,
        temperature=settings.agent_temperature,
    )


def format_agent_context(schema_summary: str, question: str) -> Dict[str, Any]:
    """
    Assemble prompt context shared between agents.
    
    This helper function formats the schema and question into a dictionary
    that can be used in agent prompts.
    
    Args:
        schema_summary: Human-readable schema description
        question: User's question
        
    Returns:
        Dictionary with schema and question keys
    """
    return {
        "schema": schema_summary,
        "question": question,
    }


class SupervisorAgent:
    """
    Supervisor agent that analyzes questions and coordinates with coding agent.
    
    The supervisor agent uses ministral-3:3b model to:
    1. Analyze user questions to determine if SQL queries are needed
    2. Synthesize natural language answers from query results
    
    This agent acts as the orchestrator, making decisions about when
    to query the database and how to present results to users.
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize the supervisor agent with Ollama LLM.
        
        Args:
            config: AgentConfig with model and connection settings
        """
        # Initialize ChatOllama with configuration
        self.llm = ChatOllama(
            base_url=config.host,
            model=config.model,
            temperature=config.temperature,
        )
        
        # System prompt that defines the agent's role and behavior
        self.system_prompt = """You are a supervisor agent in an Agentic RAG system. 
Your role is to:
1. Analyze user questions to understand what data is needed
2. Determine if a SQL query is required to answer the question
3. Coordinate with the coding agent to generate and execute queries
4. Synthesize final answers from query results

When you receive a question:
- If it requires data analysis (aggregations, filtering, grouping), respond with "NEED_QUERY"
- If it's a simple question that doesn't need data, respond with "NO_QUERY"
- Always be thorough and precise in understanding the question

After receiving query results, synthesize a clear, natural language answer."""

    def analyze_question(self, question: str, schema_info: str) -> str:
        """
        Analyze the question and determine if SQL query is needed.
        
        This method sends the question and schema information to the LLM
        and asks it to determine if a database query is required.
        
        Args:
            question: User's natural language question
            schema_info: Human-readable schema description
            
        Returns:
            String response from the LLM indicating "NEED_QUERY" or "NO_QUERY"
            along with an explanation
        """
        # Construct the prompt with schema and question
        prompt = f"""Schema information:
{schema_info}

User question: {question}

Does this question require querying the database? Respond with either "NEED_QUERY" or "NO_QUERY" followed by a brief explanation."""
        
        # Build message list with system prompt and user question
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        
        # Invoke the LLM and return the response
        response = self.llm.invoke(messages)
        return response.content

    def verify_query(self, query: str, schema_info: str) -> Dict[str, Any]:
        """
        Verify the validity of a generated SQL query against the schema.
        
        This method asks the LLM to act as a critic, ensuring that all columns
        used in the query actually exist in the referenced tables.
        
        Args:
            query: The generated SQL query to verify
            schema_info: Schema information
            
        Returns:
            Dictionary with 'valid' (bool) and 'feedback' (str)
        """
        prompt = f"""Schema information:
{schema_info}

Generated SQL Query:
{query}

Verify this query. Check specifically if:
1. All columns used in the query actually exist in their respective tables.
2. Join keys exist in BOTH tables participating in the join.
3. There are no "hallucinated" columns (e.g. using request_id on a table that doesn't have it).

Return a JSON object with this exact format:
{{
    "valid": boolean,
    "feedback": "string explaining the error if invalid, or 'Looks good' if valid"
}}
"""
        messages = [
            SystemMessage(content="You are a strict SQL Code Reviewer. Your job is to catch column hallucinations and correct them."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            # Basic cleanup if the LLM wraps JSON in markdown
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            import json
            return json.loads(content.strip())
        except Exception as e:
            # Fallback if parsing fails - assume valid to avoid getting stuck
            return {"valid": True, "feedback": f"Verification failed: {e}"}

    def synthesize_answer(
        self, question: str, query: str, results: list[Dict[str, Any]], schema_info: str
    ) -> str:
        """
        Synthesize a natural language answer from query results.
        
        This method takes the original question, the SQL query that was executed,
        and the results, then asks the LLM to create a clear, natural language
        answer that addresses the user's question.
        
        Args:
            question: Original user question
            query: SQL query that was executed
            results: List of dictionaries containing query results
            schema_info: Schema information for context
            
        Returns:
            Natural language answer synthesized from the results
        """
        # Limit results to first 50 rows to avoid prompt size issues
        results_str = str(results[:50])
        if len(results) > 50:
            results_str += f"\n... (showing first 50 of {len(results)} rows)"
        
        # Construct prompt with all relevant information
        prompt = f"""Schema information:
{schema_info}

User question: {question}

SQL query executed:
{query}

Query results:
{results_str}

Based on these results, provide a clear, thorough, and precise answer to the user's question. 
Include specific numbers, trends, and insights from the data. If the results are empty, explain why."""
        
        # Build message list and invoke LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def detect_complexity(self, question: str, schema_info: str) -> Dict[str, Any]:
        """
        Analyze if a question is complex (multi-part) and should be decomposed.
        
        A question is considered complex if it can be broken down into multiple
        single queries that each require only a few variables and simple joins.
        
        Args:
            question: User's natural language question
            schema_info: Human-readable schema description
            
        Returns:
            Dictionary with:
            - is_complex: bool indicating if decomposition is needed
            - sub_questions: list of simpler sub-questions (empty if not complex)
            - original_question: the original question preserved for aggregation
        """
        prompt = f"""Schema information:
{schema_info}

User question: {question}

Analyze this question to determine if it should be decomposed into simpler sub-questions.

A question IS COMPLEX if:
- It asks for multiple distinct metrics (e.g., "volume AND growth AND total value")
- Each metric could be answered by a simple SQL query with few columns and simple joins
- Answering everything in one query would require medium to complex UNION or multiple aggregations

A question is SIMPLE if:
- It asks for one metric or closely related metrics that naturally go together
- It can be answered with a single straightforward SQL query

If COMPLEX, break it down into sub-questions where each:
- Asks for ONE specific metric or closely related metrics
- Can be answered with a simple SELECT with basic aggregation
- Preserves any time period or filtering context from the original question

Return a JSON object with this exact format:
{{
    "is_complex": boolean,
    "sub_questions": ["sub-question 1", "sub-question 2", ...] (empty array if not complex),
    "reasoning": "brief explanation of why this is/isn't complex"
}}"""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            # Sanitize control characters that break JSON parsing
            import re
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', content.strip())
            
            import json
            result = json.loads(content)
            result["original_question"] = question
            return result
            
        except Exception as e:
            # If parsing fails, treat as simple question
            return {
                "is_complex": False,
                "sub_questions": [],
                "original_question": question,
                "reasoning": f"Failed to analyze complexity: {e}"
            }

    def aggregate_results(
        self, 
        original_question: str, 
        sub_results: list[Dict[str, Any]], 
        schema_info: str
    ) -> str:
        """
        Combine results from multiple sub-questions into a cohesive final answer.
        
        This method takes all the partial results from sub-question processing
        and synthesizes them into a comprehensive answer that addresses the
        original user question.
        
        Args:
            original_question: The original user question (before decomposition)
            sub_results: List of dictionaries, each containing:
                - sub_question: The sub-question that was asked
                - query: SQL query that was executed
                - results: Query results as list of dicts
                - partial_answer: (optional) Answer for this sub-question
            schema_info: Schema information for context
            
        Returns:
            Comprehensive natural language answer combining all sub-results
        """
        # Format sub-results for the prompt
        sub_results_formatted = []
        for i, sr in enumerate(sub_results, 1):
            formatted = f"""
--- Sub-question {i} ---
Question: {sr.get('sub_question', 'N/A')}
SQL Query: {sr.get('query', 'N/A')}
Results: {str(sr.get('results', []))[:500]}"""  # Limit result size
            sub_results_formatted.append(formatted)
        
        all_sub_results = "\n".join(sub_results_formatted)
        
        prompt = f"""Schema information:
{schema_info}

ORIGINAL USER QUESTION: {original_question}

The question was decomposed into sub-questions, and here are all the results:
{all_sub_results}

Based on ALL these results, provide a comprehensive answer that:
1. Addresses the original question completely
2. Presents all the data in a clear, organized format
3. If the question asked for a "recap" or "table", format the answer as a structured table
4. Include specific numbers and trends from each sub-question's results
5. Highlight any notable patterns or insights across the data

Synthesize a single, cohesive answer that combines all the partial results."""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class CodingAgent:
    """
    Coding agent that generates SQL queries from natural language questions.
    
    The coding agent uses qwen2.5-coder:7b model, which is specialized for
    code generation. It translates natural language questions into precise
    SQL queries that can be executed against the DuckDB database.
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize the coding agent with Ollama LLM.
        
        Args:
            config: AgentConfig with model and connection settings
        """
        # Initialize ChatOllama with configuration
        self.llm = ChatOllama(
            base_url=config.host,
            model=config.model,
            temperature=config.temperature,
        )
        
        # System prompt that defines the agent's role as a SQL generator
        self.system_prompt = """You are a coding agent specialized in generating SQL queries for DuckDB.

BEFORE WRITING ANY SQL, FOLLOW THIS CHECKLIST:

STEP 1 - Identify Required Columns:
- List ALL columns you need for SELECT, WHERE, JOIN, and GROUP BY
- For EACH column, write down WHICH TABLE it belongs to (check schema!)

STEP 2 - Verify Column Existence:
- For EVERY column you plan to use, CONFIRM it exists in the schema
- The schema shows: table_name (column1 (TYPE), column2 (TYPE), ...)
- If a column is NOT listed for a table, you CANNOT use it on that table

STEP 3 - Determine Table Strategy:
- If ALL required columns are in ONE table → Use only that table (NO JOIN)
- If you need columns from multiple tables → Plan the JOIN path

STEP 4 - JOIN Rules (ONLY if joining):
- Check "Detected Relationships" section for valid join columns
- Use ONLY the FIRST column listed in the relationship (the primary key)
- Example: "Table A and Table B linked by: trip_form_id, is_test" → use ONLY trip_form_id
- JOIN ON clause must have EXACTLY ONE condition: t1.key = t2.key
- NEVER chain conditions with AND in JOIN ON

STEP 5 - Write the Query:
- Use table aliases (t1, t2) to avoid ambiguity
- QUALIFY every column with its alias (t1.column_name)
- Double-check: Is each column actually in the table you're referencing?

COMMON MISTAKES TO AVOID:
- Using request_id on a table that doesn't have it
- Using trip_form_id on a table that doesn't have it  
- Assuming all tables have the same columns (they don't!)
- Using profile_segment without checking which table has it (only trip_forms has it)

General SQL Rules:
- For date filtering, use DATE_TRUNC or EXTRACT functions
- Always use GROUP BY when using aggregations
- Use ORDER BY for meaningful result ordering
- Return ONLY the SQL query, no explanations or markdown"""

    def generate_query(self, question: str, schema_info: str, feedback: Optional[str] = None, previous_query: Optional[str] = None) -> str:
        """
        Generate a SQL query based on the question and schema.
        
        This method sends the question and schema to the LLM and asks it
        to generate a SQL query that will answer the question.
        
        Args:
            question: User's natural language question
            schema_info: Human-readable schema description
            feedback: Optional feedback from a previous failed attempt
            previous_query: The failed query string from the previous attempt
            
        Returns:
            SQL query string (cleaned of markdown formatting if present)
        """
        # Construct prompt with schema and question
        prompt_content = f"""Schema information:
{schema_info}

User question: {question}
"""
        
        # Add feedback if provided (Self-Correction Loop)
        if feedback:
            prompt_content += f"""
=== CORRECTION REQUIRED ===
Your previous query FAILED with this error:
{feedback}
"""
            if previous_query:
                prompt_content += f"""
FAILED QUERY:
{previous_query}

IMPORTANT FIX INSTRUCTIONS:
1. The error says a column does NOT EXIST in that table. DO NOT USE THAT COLUMN.
2. Check the "Detected Relationships" section to find the CORRECT join columns.
3. If the tables you're joining don't share the column you used, find the column they DO share.
4. Consider: maybe you don't need a JOIN at all? Check if all needed columns are in ONE table.
5. Look at the relationship: "requests and trip_forms are linked by: trip_form_id" - use trip_form_id NOT request_id.
"""

        prompt_content += "\nGenerate a SQL query to answer this question. Return ONLY the SQL query, no explanations or markdown code blocks."
        
        # Build message list and invoke LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt_content),
        ]
        
        response = self.llm.invoke(messages)
        query = response.content.strip()
        
        # Clean up markdown code blocks if the LLM included them
        # Some LLMs wrap code in ```sql ... ``` blocks
        if query.startswith("```sql"):
            query = query[6:]
        if query.startswith("```"):
            query = query[3:]
        if query.endswith("```"):
            query = query[:-3]
        
        return query.strip()
