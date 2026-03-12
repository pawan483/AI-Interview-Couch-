import json
import random
import gradio as gr
from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])

with open("ai_interview_dataset.json") as f:
    interview_data = json.load(f)

categories = ["All"] + sorted(list(set(q["category"] for q in interview_data)))

current_question = None

def generate_question(category):

    global current_question

    if category == "All":
        filtered = interview_data
    else:
        filtered = [q for q in interview_data if q["category"] == category]

    q = random.choice(filtered)

    current_question = q["question"]

    return current_question
    
def evaluate_answer(answer):

    prompt = f"""
    You are an AI interviewer.

    Question: {current_question}
    Candidate Answer: {answer}

    Evaluate the answer.

    Return:
    Score (1-10)
    Strengths
    Weaknesses
    Improved Answer
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content


with gr.Blocks() as demo:

    gr.Markdown("# 🤖 AI Interview Coach")
    gr.Markdown("Practice AI / ML / LLM Interviews")

    with gr.Row():

        with gr.Column():

            category_dropdown = gr.Dropdown(
                categories,
                label="Select Category",
                value="All"
            )

            generate_btn = gr.Button("Generate Question")

            question_box = gr.Textbox(
                label="Interview Question",
                lines=4,
                interactive=False
            )

        with gr.Column():

            answer_box = gr.Textbox(
                label="Your Answer",
                lines=8
            )

            submit_btn = gr.Button("Evaluate Answer")

            feedback_box = gr.Textbox(
                label="AI Feedback",
                lines=10
            )

    generate_btn.click(
        generate_question,
        inputs=category_dropdown,
        outputs=question_box
    )

    submit_btn.click(
        evaluate_answer,
        inputs=answer_box,
        outputs=feedback_box
    )

demo.launch()
    
