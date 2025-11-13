# 📈 Agentic Data Analyst

**Your AI-Powered Interactive Data Analyst — Built with LangChain, Streamlit, and GROQ**

---

## 🚀 Overview

Agentic Data Analyst is an **AI-driven analytics assistant** that lets you upload CSV or Excel files, chat with your data in natural language, and generate instant insights — including **statistical analysis, data summaries, and visualizations**.

Built using:

- 🧠 **LangChain** for LLM orchestration
- ⚡ **GROQ LLaMA-3** models for fast, reasoning-based responses
- 📊 **Streamlit** for a rich interactive UI
- 🐼 **Pandas / Seaborn / Matplotlib** for real data analysis and visualization

---

## 🧰 Features

### 📤 File Upload Support

- Upload CSV or Excel files for instant analysis
- Automatic schema and data-type detection
- Cleans “Unnamed” index columns automatically

### 💬 Natural Language Chat

- Ask questions about your dataset (e.g. _“What’s the average salary?”_)
- Get Python-powered responses and explanations
- AI automatically writes and executes `pandas` or `matplotlib` code

### 🔍 Smart Analysis

- Get statistical summaries, missing value reports, and top values
- Perform aggregations, filtering, and group-by queries
- Visualize data via histograms, bar charts, heatmaps, and more

### 🎨 Visual Insights

- Automatically generates **Seaborn/Matplotlib** charts with proper labels and titles
- Plots appear directly in the chat

### 🧠 Chat Memory

- Remembers conversation history for contextual chat
- Auto-resets when a new dataset is uploaded
- Manual “Clear Chat History” option available

---

## 🖥️ Interface Preview

📈 Agentic Data Analyst

──────────────────────────

[Upload File] ← CSV or Excel
[Enter GROQ API Key]

──────────────────────────

💬 Chat Example:
User: Show me missing values
Assistant: The following columns contain nulls:
<CODE>
result = df.isnull().sum()
print(result[result > 0])
</CODE>

──────────────────────────

---

## ⚙️ Installation & Setup

### 🪜 Clone the Repository

```bash
git clone https://github.com/yourusername/agentic-data-analyst.git
cd agentic-data-analyst
```
