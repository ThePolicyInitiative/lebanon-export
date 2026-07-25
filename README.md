# Lebanon Industrial Export Dashboard + Export Agent

This Streamlit project displays the original industrial-export dashboard and places the Export Agent in the left sidebar.

## Chatbot improvements

- Complete dashboard database is queried before the older entity parser.
- Common export questions work without an external model key.
- Local deterministic query planning covers totals, trends, rankings, products, markets, sectors, bilateral exports, shares, RCA, PCI, HHI, EXPY, market size, unrealized potential, entry/exit, performance and data coverage.
- Questions outside the local rules use safe read-only SQL planning when Groq or the silent OpenRouter fallback is configured.
- SQL answers use stable sidebar-friendly formatting and are never rewritten by a model.
- Multi-market and multi-sector comparisons are handled directly.
- Bilateral combinations with no recorded exports return an explicit zero rather than an ambiguous missing-record message.
- Trend and comparison answers include exact calculated summaries.

## Complete dashboard-data access

The database was generated from every numerical/business dataset embedded in `dashboard_component/dashboard.html`, including:

- Annual export totals and real values, 2018–2025
- 882 HS6 products with yearly exports, RCA, PCI, market reach, growth and potential
- 187 destination markets with yearly exports, product counts, HHI, EXPY, performance and potential
- 16 sectors with yearly values, shares, RCA, PCI and potential
- 83,676 product–market–year observations
- 16,693 exact unrealized-potential product–market pairs
- 307,368 HS6 destination-market-size observations
- Similar markets, entry/exit, TOPSIS groups, methodology and source metadata
- Lossless raw embedded records

## Header layout

The wrapper corrects the original inline fixed widths using valid CSS property names. Logos and controls remain in one horizontal row at desktop widths, shrink at narrower widths, and do not overlap. The top bar scrolls normally and is not fixed.

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
