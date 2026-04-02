import streamlit as st
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from dotenv import load_dotenv
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
 
# --- LOAD CONFIG ---
load_dotenv()
TENANT_ID = "40c1b80f-7071-4cf6-8a06-cda221ff3f4d"
TENANT_SCHEMA = f"tenant_{TENANT_ID}"
DB_CONFIG = {
   "host": os.getenv("DB_HOST"),
   "dbname": os.getenv("DB_NAME"),
   "user": os.getenv("DB_USER"),
   "password": os.getenv("DB_PASSWORD"),
   "port": os.getenv("DB_PORT", 5432),
}
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)
 
# --- RAG CONFIG ---
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
INDEX_FILE = "schema_index.faiss"
CHUNKS_FILE = "schema_chunks.json"
 
# --- BUILD OR LOAD SCHEMA INDEX ---
def build_schema_index():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = %s
            """, (TENANT_SCHEMA,))
            rows = cur.fetchall()
 
            cur.execute("""
                SELECT
                    tc.table_name AS source_table,
                    kcu.column_name AS source_column,
                    ccu.table_name AS target_table,
                    ccu.column_name AS target_column
                FROM
                    information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                WHERE constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s;
            """, (TENANT_SCHEMA,))
            fk_rows = cur.fetchall()
 
    if not rows:
        raise ValueError(f"No tables found in schema '{TENANT_SCHEMA}'.")
 
    table_docs = {}
    for table, col, dtype in rows:
        table_docs.setdefault(table, []).append(f"{col} ({dtype})")
 
    relationships = []
    for src_table, src_col, tgt_table, tgt_col in fk_rows:
        relationships.append(f"{src_table}.{src_col} → {tgt_table}.{tgt_col}")
 
    chunks = []
    for table, cols in table_docs.items():
        rels = [r for r in relationships if r.startswith(table)]
        chunk = f"{table}: {', '.join(cols)}"
        if rels:
            chunk += f"\nRelationships: {', '.join(rels)}"
        chunks.append(chunk)
 
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)
    return index, chunks
 
def load_schema_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE) as f:
            chunks = json.load(f)
        return index, chunks
    return build_schema_index()
 
# Lazy load - initialize as None, load on first use
index = None
chunks = None

def get_or_load_index():
    global index, chunks
    if index is None or chunks is None:
        try:
            index, chunks = load_schema_index()
        except Exception as e:
            st.error(f"Failed to load schema index: {str(e)}")
            raise
    return index, chunks
 
# --- RAG: Get relevant tables ---
def get_relevant_tables(question, top_k=7):
    idx, chks = get_or_load_index()
    q_emb = model.encode([question])
    D, I = idx.search(np.array(q_emb), top_k)
    return [chks[i] for i in I[0]]
 
# --- DB EXECUTION WITH FRIENDLY ERROR HANDLING ---
def run_sql(query):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                return {"success": True, "data": cur.fetchall()}
    except psycopg2.Error as e:
        return {"success": False, "error": str(e)}
 
# --- GENERATE SQL ---
def generate_sql(user_question, conversation_history):
    relevant_schema = "\n".join(get_relevant_tables(user_question))
 
    context = ""
    if conversation_history:
        context = "Previous conversation:\n"
        for i, (q, a) in enumerate(conversation_history[-3:]):
            context += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"
 
    prompt = f"""
    You are an expert PostgreSQL assistant for a multi-tenant Workdesk database.
    You are a translator for natural language -> SQL.
    All tables are inside the schema "{TENANT_SCHEMA}".
    Always use schema-qualified table names in the format "{TENANT_SCHEMA}"."table_name".
    Do NOT use tenant_id filters — data isolation is by schema.
 
    When filtering by project.title or similar text column:
    - Use case-insensitive matching with ILIKE
    - Ignore spaces and hyphens in matching:
      REPLACE(REPLACE(LOWER(text_column), ' ', ''), '-', '') ILIKE '%' || REPLACE(REPLACE(LOWER(search_value), ' ', ''), '-', '') || '%'
 
    STRICT RULES:
    1. ALWAYS use "{TENANT_SCHEMA}"."table_name" format
    2. Use proper JOIN syntax
    3. Prefer filtering by ID columns when possible
    4. Return ONLY the SQL query, no explanations
    5. End with semicolon
    6. Escape single quotes in values
    7. Match closest table/column if unsure
    8. If the user didn't reference an object explicitly, infer the most likely one.
    9. ISO format for dates
    10. ILIKE for text searches
    11. When a column is marked as (FK → table.column), always use it to JOIN with the referenced table.
    12. If the user asks a question involving attributes from multiple tables, infer the join path using FK references.
 
 
    BUT:
    If the user explicitly asks to "summarize", "overview", "explain", or "give insights", then DO NOT generate SQL.
    Instead, provide a concise natural language summary.
    If the user’s question requires both data and summary:
    First generate the SQL query, then provide a short natural language answer.
 
    Example:
    SELECT p.id, p.name, c.name AS category
    FROM "{TENANT_SCHEMA}"."project" p
    JOIN "{TENANT_SCHEMA}"."project_category" c
    ON p.category_id = c.id;
 
    -- Example: joining multiple tables (project, issue, priority, status, user)
    SELECT p.id AS project_id,
           p.name AS project_name,
           i.id AS issue_id,
           i.title AS issue_title,
           pri.level AS priority_level,
           s.state AS status,
           u.full_name AS assigned_to,
           i.created_at
    FROM "{TENANT_SCHEMA}"."project" p
    JOIN "{TENANT_SCHEMA}"."issue" i
        ON p.id = i.project_id
    JOIN "{TENANT_SCHEMA}"."priority" pri
        ON i.priority_id = pri.id
    JOIN "{TENANT_SCHEMA}"."status" s
        ON i.status_id = s.id
    LEFT JOIN "{TENANT_SCHEMA}"."user" u
        ON i.assigned_to = u.id
    WHERE REPLACE(REPLACE(LOWER(p.name), ' ', ''), '-', '')
        ILIKE '%' || REPLACE(REPLACE(LOWER('search_value'), ' ', ''), '-', '') || '%';
 
    Summarization (no SQL)
    Question: "Summarize the issues in Sample Project"
    Answer: "The Sample project currently has several issues, most marked as 'Open' with high priority. The majority are related to integration and testing."
 
    SQL + Summary
    Question: "Show me the issues in Sample project and summarize them"
    SQL:
    SELECT i.id, i.title, s.name AS status, pr.name AS priority, p.name AS project
    FROM "{TENANT_SCHEMA}"."issue" i
    JOIN "{TENANT_SCHEMA}"."status" s ON i.status_id = s.id
    JOIN "{TENANT_SCHEMA}"."priority" pr ON i.priority_id = pr.id
    JOIN "{TENANT_SCHEMA}"."project" p ON i.project_id = p.id
    WHERE REPLACE(REPLACE(LOWER(p.name), ' ', ''), '-', '')
    ILIKE '%' || REPLACE(REPLACE(LOWER('Sample'), ' ', ''), '-', '') || '%';
 
    Answer: "Most of the Sample project issues are open, with high priority. A few are resolved and marked low priority."
 
 
    {context}
 
    Use ONLY these tables/columns:
    {relevant_schema}
 
    Current Question: {user_question}
    """
 
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )
 
    return response.choices[0].message.content.strip()
 
# --- Check if response looks like SQL ---
def is_sql_query(text: str) -> bool:
    sql_keywords = ("select", "with", "insert", "update", "delete")
    return text.strip().lower().startswith(sql_keywords)
 
# --- Dynamic clarification helper ---
def ask_for_clarification(user_question, error_reason=None, result_data=None):
    clarification_prompt = f"""
    The user asked: "{user_question}"
 
    {f"The SQL failed with error: {error_reason}" if error_reason else ""}
    { "The SQL returned no results." if result_data == [] else "" }
 
    Your job:
    - Do NOT provide SQL.
    - Politely ask the user for more details to clarify their intent.
    """
 
    clarification_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": clarification_prompt}],
        temperature=0.3
    )
    return clarification_response.choices[0].message.content.strip()
 
# --- STREAMLIT APP ---
st.title("Go Desk AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Workdesk data assistant. How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Workdesk data"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # --- build conversation history for context ---
            conversation_history = []
            for i in range(0, len(st.session_state.messages)-1, 2):
                if i+1 < len(st.session_state.messages):
                    conversation_history.append(
                        (st.session_state.messages[i]["content"],
                         st.session_state.messages[i+1]["content"])
                    )

            # --- generate SQL from user question ---
            response = generate_sql(prompt, conversation_history)

            if is_sql_query(response):
                message_placeholder.markdown("Analysing...")
                result = run_sql(response)

                if result.get("success"):
                    if result.get("data"):
                        # ✅ got valid results → summarize
                        summary_prompt = f"""
                        Question: {prompt}
                        SQL Result: {result['data']}
                        Provide a concise natural language answer.
                        """
                        summary_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "system", "content": summary_prompt}],
                            temperature=0.1
                        )
                        answer = summary_response.choices[0].message.content
                        full_response = answer  # Changed: Only show the answer, not the query
                    else:
                        # ✅ query succeeded but returned no rows → clarify
                        full_response = ask_for_clarification(prompt, result_data=result.get("data"))
                else:
                    # --- AI Self-correction if SQL failed ---
                    correction_prompt = f"""
                    The following SQL query failed:

                    {response}

                    Error message:
                    {result.get('error')}

                    Please fix the SQL query so it runs successfully,
                    following the same formatting rules as before.
                    Only return the corrected SQL.
                    """
                    correction_response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "system", "content": correction_prompt}],
                        temperature=0
                    )
                    corrected_sql = correction_response.choices[0].message.content.strip()

                    if is_sql_query(corrected_sql):
                        retry_result = run_sql(corrected_sql)
                        if retry_result.get("success") and retry_result.get("data"):
                            summary_prompt = f"""
                            Question: {prompt}
                            SQL Result: {retry_result['data']}
                            Provide a concise natural language answer.
                            """
                            summary_response = client.chat.completions.create(
                                model="gpt-4-turbo",
                                messages=[{"role": "system", "content": summary_prompt}],
                                temperature=0.1
                            )
                            answer = summary_response.choices[0].message.content
                            full_response = answer  # Changed: Only show the answer, not the query
                        else:
                            # ✅ still failed after correction → clarify
                            full_response = ask_for_clarification(prompt, error_reason=retry_result.get("error"))
                    else:
                        # ✅ correction not SQL → clarify
                        full_response = ask_for_clarification(prompt, error_reason=result.get("error"))
            else:
                # ✅ if generate_sql didn't return SQL at all → clarify
                full_response = ask_for_clarification(prompt)

        except Exception as e:
            full_response = f"Something went wrong: {str(e)}"

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
 
 
