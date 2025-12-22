🌟 Wonderful Workflow Corpus
============================

This repository is a collection of reusable workflows, each defined by a `workflow.json` (workflow logic) and a `tools.json` (tool definitions). All workflows can be executed through the universal script `execute_workflow.py` by passing in a goal string written in natural language.

The corpus includes both simple workflows, such as **Arxiv Daily Digest**, **Recipe Generator**, or **Tetris Game**, which can be run directly with a short goal, and advanced workflows like **Invest (Stock Analysis)**, which combine file discovery, data loading, and multi-stage analysis. This way the corpus supports everything from quick experiments to more complex analysis tasks.  

------------------------------------------------------------

🚀 Quick Start
--------------

1. Install dependencies:
   pip install -r requirements.txt

2. Make sure your OpenAI API Key is set either as an environment variable or in a .env file:
   OPENAI_API_KEY=sk-xxxxxx

3. Execute a workflow (general command):
   ```bash
   python execute_workflow.py `
     --workflow <path-to-workflow.json> `
     --goal "<your-goal>" `
     --output <result-file>
   ```

------------------------------------------------------------

📂 Example Workflows
--------------------

1. Arxiv Daily Digest
   Recommend the latest academic papers from arXiv based on keywords.

   ```bash
   python Wonderful_workflow_corpus/execute_workflow.py `
     --workflow Wonderful_workflow_corpus/arxiv_daily_digest/workflow.json `
     --goal "Please recommend the latest papers on multi-agent systems in natural language processing. Summarize each paper in 3 sentences and provide direct arXiv links." `
     --output arxiv_digest.md
   ```

- Arxiv Daily Digest workflow was contributed by Shiyang Liang: victor.liang28@outlook.com & [@Haoyang Li](https://github.com/Oxygennnnn).

------------------------------------------------------------

2. Recipe Generator
   Generate a recipe based on user preferences.

   ```bash
   python Wonderful_workflow_corpus/execute_workflow.py `
     --workflow Wonderful_workflow_corpus/recipe_meal_plan/workflow.json `
     --goal "Generate a healthy vegetarian dinner recipe that uses tofu, broccoli, and garlic. Include step-by-step instructions and estimated cooking time." `
     --output recipe.md
   ```

   - Recipe Generator workflow was contributed by [@xyq116](https://github.com/xyq116).

------------------------------------------------------------

3. Tetris Game Generator
   Generate HTML/CSS/JavaScript code for a browser-based Tetris game.

   ```bash
   python Wonderful_workflow_corpus/execute_workflow.py `
     --workflow Wonderful_workflow_corpus/tetris_game/workflow.json `
     --goal "Generate a playable Tetris game with scoring, level progression, and keyboard controls. The game should run in any modern browser." `
     --output tetris.html
   ```

   Open tetris.html in a browser to play.
   - Tetris Game Generator workflow was contributed by [@fangjy6](https://github.com/fangjy6).

------------------------------------------------------------

4. Travel Recommendation Generator
   Generate a travel recommendation based on user preferences.

   ```bash
   python Wonderful_workflow_corpus/execute_workflow.py `
     --workflow Wonderful_workflow_corpus/travel_recommendation/workflow.json `
     --goal "Plan a 7-day trip for 2 adults from San Francisco (SFO) to Tokyo on 2025-10-12 to 2025-10-18 with a total budget of USD 3,200. Prefer a 4-star hotel in Shinjuku or Shibuya with breakfast, fast Wi-Fi, and a gym (≤220 USD/night). Round-trip flights; local metro + walking (Suica). Dining: ramen, sushi, izakaya (mid-range). Interests: culture, contemporary art, historic neighborhoods, light outdoor; medium comfort; no group tours." `
     --output travel_recommendation.md
   ```
   
   - Travel Recommendation Generator workflow was contributed by [@xyq116](https://github.com/xyq116).

------------------------------------------------------------

5. Feng Shui Advisor
   Generate personalized home Feng Shui (风水) analysis and improvement recommendations based on user descriptions, producing detailed room-by-room layout advice and practical remedies in the same language as the input.

   ```bash
   python Wonderful_workflow_corpus/execute_workflow.py `
     --workflow Wonderful_workflow_corpus/Fengshui_analysis/workflow.json `
     --goal "I live in a two-bedroom apartment on the 12th floor facing southeast. The living room and kitchen share an open layout, and my bedroom door faces directly toward the main entrance. Lately, I’ve felt low in energy and often have trouble sleeping. There’s also clutter near the balcony area where I usually work. I’d like to improve my sleep quality and create a more harmonious and productive atmosphere at home, but I can only make small, low-cost adjustments because it’s a rented apartment. Please analyze possible Feng Shui issues and give me practical suggestions for rearranging furniture, improving energy flow, and creating better balance." `
     --output Fengshui_analysis.md
   ```
   
   - Feng Shui Advisor workflow was contributed by [@xyq116](https://github.com/xyq116).

------------------------------------------------------------

📊 Advanced Workflow
--------------------

6. Example: Stock Analysis
Run the full pipeline with:

```bash
python Wonderful_workflow_corpus/invest/stock_analysis.py
```
- Stock Analysis workflow was contributed by [@bruuuuuceliu](https://github.com/bruuuuuceliu) & [@yuanxiaochenAC](https://github.com/yuanxiaochenAC).

The process will:

1. Create required folders.
2. Fetch stock data (if not already available).
3. Generate graphs (if not already available).
4. Execute the stock analysis workflow.
5. Save a Markdown report and an HTML report.

------------------------------------------------------------

⚠️ Notes
- `stock_analysis.py` automatically selects the workflow file based on the operating system: `workflow_windows.json` for Windows users, and `workflow.json` for macOS/Linux users.
- Windows users: workflows use `cmd /c dir /a /b /s "<PATH>"` for file discovery.
- macOS/Linux users: workflows use `find "<PATH>" -type f -print` for file discovery.
- If cross-platform support is needed, ensure workflow prompts strictly differentiate between OS commands.

------------------------------------------------------------

✨ Extending the Corpus
-----------------------

You can add new workflows by:
1. Writing a workflow.json and tools.json in a new subdirectory.
2. Running them with the same executor:
   python execute_workflow.py --workflow path/to/new/workflow.json --goal "<goal>" --output result.md

------------------------------------------------------------
