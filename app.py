from flask import Flask, render_template, request, jsonify
import cohere
from dotenv import load_dotenv
import os
from datetime import datetime
import re

load_dotenv()

app = Flask(__name__)

# Configure Cohere
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Template format
SUMMARY_TEMPLATE = """
=== DATASNIPER.LOG ===

🧠Topic:
{query}

📘Brief Info:
{concept}

💡Applications:
{applications}

🧪Examples:
{examples}

🛠️Tools:
{tools}

⚠️Limitations:
{limitations}
"""

def generate_summary(query):
    prompt = f"""
You are a structured AI researcher assistant.

Given the query: "{query}", generate a structured response with EXACTLY these 6 headings in this order:

Brief Concept:
Applications:
Examples:
Tools:
Limitations:

Instructions:
- Use only one short paragraph per section.
- In Applications, Tools, and Limitations sections, use 3–5 clear bullet points instead of a paragraph.
- Do not add introductions or summaries.
- After the 'Limitations' section, end the response with this phrase exactly: 'Tell me more about what you want to know next!'

Output must be clean, minimal, and consistently formatted.
"""

    try:
        response = co.chat(
            model="command-r",
            message=prompt,
            temperature=0.7,
        )
        raw = response.text
        print("=== RAW RESPONSE ===")
        print(response.text)


        sections = {
            "concept": extract_section(raw, ["brief concept", "concept"]),
            "applications": extract_section(raw, ["applications"]),
            "examples": extract_section(raw, ["examples"]),
            "tools": extract_section(raw, ["tools", "libraries"]),
            "limitations": extract_section(raw, ["limitations"]),
            "next_steps": extract_section(raw, ["next to explore", "further reading"]),
        }

        formatted = SUMMARY_TEMPLATE.format(query=query, **sections)
        return formatted.strip()
    except Exception as e:
        raise RuntimeError(f"Cohere Error: {e}")


def extract_section(text, keywords):
    text = text.lower()
    for key in keywords:
        pattern = rf"{key}[:\-]?\s*(.*?)(?=\n\s*\w+[:\-]|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result = match.group(1).strip().capitalize()
            return result if result else "Not available."
    return "Not available."


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is missing"}), 400

    try:
        summary = generate_summary(query)

        with open("queries.log", "a", encoding="utf-8") as f:
            f.write(f"\n\n=== {datetime.now()} ===\nQuery: {query}\n\n{summary}\n")

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)
