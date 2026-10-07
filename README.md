## Setup

1. Clone the repo
   \`\`\`bash
   git clone https://github.com/thisisadison/quorum
   cd quorum
   \`\`\`

2. Create a virtual environment
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   \`\`\`

3. Install dependencies
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. Set up API keys
   \`\`\`bash
   cp .env.example .env
   # then edit .env and fill in your actual API keys
   \`\`\`