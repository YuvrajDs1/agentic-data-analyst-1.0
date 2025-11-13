import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import io
import json
import re

# streamlit header
st.set_page_config(page_title="Agentic Data Analyst", page_icon="📈", layout="wide")
st.title("📈 Agentic Data Analyst")
st.markdown("*Powered by LangChain & GROQ*")

# sidebar
st.sidebar.header("⚙️ Configuration")
api_value = st.sidebar.text_input("Enter your GROQ API key", type='password')
st.sidebar.markdown("[Get a GROQ API key](https://console.groq.com/docs/models)")

model_choice = st.sidebar.selectbox("Model", options=['llama-3.1-70b-versatile', 'llama-3.1-8b-instant'], index=1)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, step=0.1)

# Require API key 
if not api_value:
    st.warning("⚠️ Please enter your API key to proceed.")
    st.stop()

st.session_state["api_key"] = api_value

# File upload
file = st.file_uploader(
    "📤 Upload a single file for data analysis (CSV or Excel)",
    type=["csv", "xlsx"], accept_multiple_files=False
)

if file is None:
    st.info("👆 Please upload a CSV or Excel file to begin analysis.")
    st.stop()

# Read file
try:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    # Clean unnecessary columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    st.success("✅ File uploaded successfully!")
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    with st.expander("📊 Dataset Preview"):
        st.dataframe(df.head(10))
        
except Exception as e:
    st.error(f"Error reading file: {str(e)}")
    st.stop()

# Initialize State
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Clear chat if new file uploaded
if "last_file" not in st.session_state or st.session_state["last_file"] != file.name:
    st.session_state["messages"] = []
    st.session_state["last_file"] = file.name

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state["messages"] = []
    st.rerun()

# Helper function

def execute_code_safely(code: str, context: dict) -> tuple:
    """Execute Python code safely and return result and any plot"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Prepare execution context
        exec_globals = {
            'df': context['df'],
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns
        }
        
        # Execute the code
        exec(code, exec_globals)
        
        # Check if a plot was created
        plot_buf = None
        if plt.get_fignums():
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plot_buf = buf
            plt.close('all')
        
        # Get any result if stored
        result = exec_globals.get('result', None)
        
        return True, result, plot_buf
    except Exception as e:
        plt.close('all')
        return False, str(e), None

def get_data_info() -> str:
    """Get comprehensive dataset information"""
    info = []
    info.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    info.append(f"\nColumns and Types:")
    for col in df.columns:
        info.append(f"  - {col}: {df[col].dtype}")
    info.append(f"\nMissing Values:")
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        info.append(f"  - {col}: {missing[col]} missing")
    if missing.sum() == 0:
        info.append("  - No missing values")
    
    info.append(f"\nNumeric Columns Summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info.append(df[numeric_cols].describe().to_string())
    
    return "\n".join(info)

# Initialize LLM
llm = ChatGroq(
    model=model_choice,
    api_key=SecretStr(st.session_state["api_key"]),
    temperature=temperature,
    max_tokens=3000
)

# Prompt
system_prompt = f"""You are an expert data analyst. You have access to a pandas DataFrame called 'df' with the following information:

DATASET INFO:
{get_data_info()}

Your job is to help users analyze this data. When users ask questions, you should:

1. **For summary/info questions**: Describe the data structure, columns, statistics
2. **For analysis questions**: Write Python code using pandas to analyze the data
3. **For visualization questions**: Write matplotlib/seaborn code to create plots

IMPORTANT INSTRUCTIONS:
- When you need to execute code, wrap it in <CODE> tags like this: <CODE>your_code_here</CODE>
- For visualizations, create complete matplotlib/seaborn plots with titles, labels, and styling
- Always use the exact column names from the dataset
- Store results in a variable called 'result' if you want to show output
- For plots, the code will automatically be executed and displayed

EXAMPLES:

User: "Show summary statistics"
Assistant: Here's a summary of the dataset:
<CODE>
result = df.describe()
print(result)
</CODE>

User: "Create a histogram of column_name"
Assistant: I'll create a histogram showing the distribution:
<CODE>
plt.figure(figsize=(10, 6))
plt.hist(df['column_name'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
plt.title('Distribution of Column Name', fontsize=14, fontweight='bold')
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
</CODE>

User: "What are the top 5 values in column_name?"
Assistant: Let me find the top 5 values:
<CODE>
result = df.nlargest(5, 'column_name')[['column_name']]
print(result)
</CODE>

Now respond to the user's question. Be conversational and explain what you're doing."""

# Chat History
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("plot"):
            try:
                st.image(msg["plot"])
            except:
                pass

# Chat input
user_input = st.chat_input("Ask me anything about your data...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input, "plot": None})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your data..."):
            try:
                # Create prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{question}")
                ])
                
                chain = prompt | llm | StrOutputParser()
                
                # Get LLM response
                response = chain.invoke({"question": user_input})
                
                print(f"LLM Response: {response}")  # Debug
                
                # Extract code blocks
                code_pattern = r'<CODE>(.*?)</CODE>'
                code_matches = re.findall(code_pattern, response, re.DOTALL)
                
                # Remove code blocks from response text
                clean_response = re.sub(code_pattern, '', response, flags=re.DOTALL)
                clean_response = clean_response.strip()
                
                # Display the text response
                if clean_response:
                    st.markdown(clean_response)
                
                # Execute any code found
                plot_data = None
                for code in code_matches:
                    code = code.strip()
                    print(f"Executing code:\n{code}")  # Debug
                    
                    success, result, plot_buf = execute_code_safely(code, {'df': df})
                    
                    if success:
                        if result is not None:
                            st.code(str(result), language='text')
                        if plot_buf:
                            st.image(plot_buf)
                            plot_data = plot_buf.getvalue()
                    else:
                        st.error(f"Error executing code: {result}")
                
                # Store message
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": clean_response if clean_response else response,
                    "plot": plot_data
                })
                
            except Exception as e:
                error_msg = f"❌ **Error:** {str(e)}"
                st.error(error_msg)
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": error_msg,
                    "plot": None
                })

