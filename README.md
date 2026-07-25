# Lebanon Industrial Export Dashboard + Universal Export Agent

This Streamlit project displays the original industrial-export dashboard and places the Export Agent in the left sidebar.

## Universal dashboard access

The agent queries the complete read-only database generated from every numerical and business dataset embedded in `dashboard_component/dashboard.html` before using any general-language fallback.

It can answer questions involving any dashboard product, sector, destination market, continent or year, including combinations of them:

- Lebanon's total and real exports, 2018–2025
- Any of 882 HS6 products, using product names by default
- Any of 187 destination markets
- Any of 16 sectors
- Product–country and sector–country exports
- Rankings, shares, counts, trends and direct comparisons
- RCA, PCI, HHI, EXPY, CAGR and export reach
- Market size and market penetration
- Actual exports versus unrealized potential
- Potential by product, destination, sector or continent
- New and exited products, nationally or within a destination
- Growth drivers by product, sector or market
- Products common to two markets or present in one but not another
- Markets buying two named products
- Correlations and averages across products, markets or sectors
- Similar markets, performance classifications, data coverage and methodology

The database includes:

- 8 annual total-export observations
- 7,056 product-year observations
- 1,496 market-year observations
- 128 sector-year observations
- 83,676 product-market-year observations
- 16,693 unrealized-potential product-market pairs
- 307,368 product-market-size observations
- Lossless raw records and source metadata from the dashboard

## Answer safeguards

- Exact dashboard figures are returned deterministically and are not rewritten by the language model.
- Product names are shown by default. HS codes appear only when explicitly requested or supplied.
- Groq is optional and primary when configured; OpenRouter is a silent fallback.
- Common and advanced dashboard questions work without an API key.
- SQL is read-only and restricted to one `SELECT`/`WITH` statement.

## Chat interface

- Times New Roman is applied throughout the chatbot.
- Ranked answers use numbered lists.
- The newest answer remains in view.
- The composer remains fixed at the bottom of the sidebar.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional environment variables:

```text
GROQ_API_KEY
GROQ_MODEL
OPENROUTER_API_KEY
OPENROUTER_MODEL
TAVILY_API_KEY
```

Upload the extracted files to the GitHub repository root and set the Streamlit main file to `app.py`.
