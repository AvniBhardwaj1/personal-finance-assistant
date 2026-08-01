Finance Assistant — Project Handoff

Overview

This repository contains an in-progress AI-powered, Streamlit-based Personalized Financial Assistant. The model (fine-tuned + quantized) and supporting code let the assistant read a local personal finance CSV and answer client-specific questions using a ReAct-style agent that can call simple tools (CSV lookup, live stock price via yfinance).

What’s included

- app.py — Streamlit app and agent wiring (LangChain + Ollama integration).
- synthetic_personal_finance_dataset.csv — local dataset used by the app (Kaggle-derived). Large; consider not committing to remote.
- Llama3-Finance-Q4_K_M.gguf — large local model file (gguf) — DO NOT push to GitHub without Git LFS or external storage.
- Modelfile — simple Dockerfile fragment referencing the local gguf model.
- current_state.txt — handover notes and roadmap.

Quick status (what’s done)

- Model: Fine-tuning and quantization completed locally. A gguf model file (Llama3-Finance-Q4_K_M.gguf) exists in the project root.
- App: A functioning Streamlit app skeleton (app.py) that:
  - Loads the CSV (pandas)
  - Provides two LangChain Tools: GetLiveStockPrice (yfinance) and GetClientProfile (CSV lookup)
  - Connects to a local Ollama-backed model using langchain_ollama.ChatOllama
  - Uses a ReAct prompt template and AgentExecutor invoke flow
- Virtualenv: A working venv included for the developer machine (not committed). Requirements are captured in requirements.txt

What’s left / recommended next steps (to pick up where you left off)

1) Avoid committing large binaries
   - The gguf model and large CSV are currently in the project folder. Do not push these to GitHub as normal files. Use Git LFS, a separate cloud bucket, or keep them local.
2) Ollama setup
   - The app expects an Ollama runtime with your fine-tuned model loaded as "my-finance-bot". To replicate on another machine:
     - Install Ollama (https://ollama.com/docs)
     - Load the gguf file: ollama import --from-file ./Llama3-Finance-Q4_K_M.gguf --model my-finance-bot
     - Confirm Ollama is running (ollama serve)
3) Streamlit UI improvements (recommended)
   - Add a client ID input/select UI (instead of relying on a chat prompt example).
   - Add charts (spending vs savings) via streamlit.pyplot or streamlit.altair_chart.
   - Ensure yfinance returns live prices in your runtime (some yfinance info keys differ by version).
4) ReAct / agent testing
   - Create unit tests for the Tools: CSV lookup and stock lookup.
   - Run interaction tests verifying the agent uses the CSV tool, then the stock tool, then synthesizes an answer.
5) Security and tokens
   - If you enable remote pushes or API access, do not commit secrets. Use environment variables or a secrets manager.

How to run locally (fresh machine)

1) Install Python 3.14 (the dev environment used here).
2) Create and activate a venv:
   python3 -m venv venv
   source venv/bin/activate
3) Install dependencies:
   pip install -r requirements.txt
4) Place the model and dataset locally (see notes above):
   - Copy Llama3-Finance-Q4_K_M.gguf into project root (or configure Ollama to point to it)
   - Ensure synthetic_personal_finance_dataset.csv is present in project root
5) Start Ollama and load model (example):
   ollama import --from-file ./Llama3-Finance-Q4_K_M.gguf --model my-finance-bot
   ollama serve
6) Run Streamlit:
   streamlit run app.py

How to push this repo to GitHub (recommended workflow)

Option A — using a Personal Access Token (manual):

1) Create a GitHub PAT with repo scope.
2) Create a private repo (example using curl):
   curl -H "Authorization: token <PAT>" \
     -d '{"name":"finance-assistant-project","private":true}' \
     https://api.github.com/user/repos
3) Add remote and push:
   git remote add origin https://github.com/<your-username>/finance-assistant-project.git
   git branch -M main
   git push -u origin main

Option B — use GitHub CLI (gh):
   gh auth login
   gh repo create finance-assistant-project --private --confirm
   git push -u origin main

Notes about large files

- If you need to store the model on GitHub, use Git LFS and be mindful of storage/bandwidth. Alternatively, keep the model in cloud storage (S3, GCS) or share via a private file share.

Contact / Handover Summary

- The fine-tuned "brain" is complete — what remains is engineering to connect it safely to the UI and to productionize the runtime (Ollama service, model hosting, streamlit UI polish).
- See current_state.txt for an extended handover (detailed dataset and step-by-step testing notes).

If you want, I can:
- Create the private GitHub repo and push (I can do that here if you provide a PAT or the remote URL), or
- Prepare everything locally (already done) and give exact push commands you can run locally.

