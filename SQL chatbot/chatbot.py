from db import get_connection
from llm_sql import generate_sql

def chatbot_response(question):

    print("USER QUESTION:", question)

    sql = generate_sql(question)

    # REMOVE SEMICOLON (Oracle Python fix)
    sql = sql.replace(";", "")

    print("GENERATED SQL:", sql)

    if not sql.lower().startswith("select"):
        return "Only SELECT queries allowed"

    con = get_connection()
    cur = con.cursor()

    try:
        cur.execute(sql)
        rows = cur.fetchall()

        if not rows:
            return "No records found"

        result = ""
        for r in rows:
            result += str(r) + "\n"

        return result

    except Exception as e:
        return str(e)