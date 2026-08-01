Contributing to Personal Finance Assistant

Thank you for your interest in contributing. This document explains how to set up a development environment, open issues, and submit pull requests.

Getting started locally

1. Clone the repository and create a virtual environment:
   git clone https://github.com/AvniBhardwaj1/personal-finance-assistant.git
   cd personal-finance-assistant
   python3 -m venv venv
   source venv/bin/activate
2. Install dependencies:
   pip install -r requirements.txt
3. Do NOT commit large model files or datasets directly. See .gitignore.
   - The gguf model (Llama3-Finance-Q4_K_M.gguf) and the dataset are intentionally kept out of the remote repository.
4. Run the app for development:
   - Start Ollama and load the model (if you have it locally):
       ollama import --from-file ./Llama3-Finance-Q4_K_M.gguf --model my-finance-bot
       ollama serve
   - Run Streamlit:
       streamlit run app.py

Reporting issues

- Please open a GitHub Issue describing the bug or feature request with steps to reproduce and expected behavior. Include logs and environment details when applicable.

Branching and pull requests

- Work on feature branches off main. Branch names should be descriptive, e.g. feature/add-streamlit-charts.
- Open a pull request and describe the change and rationale. Add screenshots if the change affects UI.
- PRs will be reviewed and merged once CI checks pass and at least one maintainer approves.

Coding style and tests

- Keep changes focused and small. Follow existing code style.
- There are no automated tests yet. Please add tests for new features when possible.

License and ownership

- This repository currently does not include a license file. By contributing, you are not granting any additional rights beyond what GitHub's default terms provide. If you want your contributions to be licensed, please open an issue to select a license.

Contact

- For questions about the model or data-handling approach, see current_state.txt and README.md.
