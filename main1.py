import streamlit as st
from agents import answer_question

# -------------------
# Page config
# -------------------
st.set_page_config(
    page_title="Book Chatbot",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Academic Book Chatbot")
st.caption("Powered by Gemini + Qdrant (Free Tier)")

# -------------------
# Session state
# -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------
# Input box
# -------------------
question = st.text_input(
    "Ask a question from the book:",
    placeholder="e.g. What are sensors in robotics?"
)

# -------------------
# Ask button
# -------------------
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer, sources = answer_question(question)

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}")

# -------------------
# Display chat
# -------------------
for chat in reversed(st.session_state.chat_history):
    st.markdown("### ❓ Question")
    st.write(chat["question"])

    st.markdown("### ✅ Answer")
    st.write(chat["answer"])

    if chat["sources"]:
        st.markdown("### 📚 Sources")
        for src in chat["sources"]:
            st.markdown(f"- {src}")

    st.divider()
