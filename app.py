from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import google.generativeai as genai
import cohere

COHERE_API_KEY = os.getenv("COHERE_API_KEY") or "your-cohere-api-key"

# Initialize Cohere
co = cohere.Client(COHERE_API_KEY)

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template("index.html")


# Format prompt based on selected style and language
def format_prompt(prompt_type, topic_or_prompt, language="english"):
    language = language.strip().lower()
    prompt_type = prompt_type.strip().lower()

    if language not in ["english", "hindi"]:
        language = "english"

    if prompt_type == "deep":
        if language == "english":
            return (
                f"Write a detailed, in-depth analysis essay of at least 400 words on the topic: \"{topic_or_prompt}\".\n"
                "Include background, key concepts, real-world applications, and important facts. "
                "Use formal tone and clear structure. Avoid repetition and general statements."
            )
        elif language == "hindi":
            return (
                f"विषय \"{topic_or_prompt}\" पर कम से कम 400 शब्दों में एक गहन विश्लेषणात्मक निबंध लिखें।\n"
                "उसका इतिहास, मुख्य तत्व, वास्तविक जीवन में उपयोग और महत्वपूर्ण तथ्य शामिल करें। "
                "भाषा औपचारिक होनी चाहिए और संरचना स्पष्ट होनी चाहिए। दोहराव और सामान्य वाक्य से बचें।"
            )

    elif prompt_type == "brief":
        if language == "english":
            return (
                f"Explain the topic \"{topic_or_prompt}\" in **no more than 3 lines**. "
                "Be concise, avoid fluff, and provide the essence of the concept only."
            )
        elif language == "hindi":
            return (
                f"विषय \"{topic_or_prompt}\" को **केवल 3 पंक्तियों में** स्पष्ट और संक्षेप में समझाइए। "
                "अनावश्यक जानकारी से बचें और केवल मुख्य बात बताएं।"
            )

    elif prompt_type == "custom":
        if language == "english":
            return topic_or_prompt
        elif language == "hindi":
            return f"इस प्रश्न का उत्तर हिंदी में दीजिए:\n{topic_or_prompt}"

    # Fallback (in case prompt_type is invalid)
    return f"Explain this in a helpful way: {topic_or_prompt}"


# Generate summary using Cohere
def generate_summary(query):
    try:
        cohere_response = co.generate(
            model='command-r-plus',
            prompt=query,
            max_tokens=2000,
            temperature=0.7,
            stop_sequences=["--END--"]
        )
        print("✅ Cohere succeeded.")
        return cohere_response.generations[0].text.strip()
    except Exception as cohere_error:
        print("❌ Cohere failed:", cohere_error)
        raise Exception("Cohere failed.")


# Summarize API
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    query = data.get("query", "").strip()
    style = data.get("style", "custom")
    language = data.get("language", "english")

    if not query:
        return jsonify({"error": "❌ Query is missing"}), 400

    try:
        final_prompt = format_prompt(style, query, language)
        summary = generate_summary(final_prompt)

        # Save to log
        with open("queries.log", "a", encoding="utf-8") as f:
            f.write(f"\n\n=== {datetime.now()} ===\nStyle: {style}\nLanguage: {language}\nQuery: {query}\n\nSummary: {summary}\n")

        return jsonify({"summary": summary})
    except Exception as e:
        print("Error while summarizing:", e)
        return jsonify({"error": str(e)}), 500


# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port)
