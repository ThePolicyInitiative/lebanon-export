# Lebanon Industrial Export Dashboard with Export Agent

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Chat interface

- The agent panel begins at the top of the sidebar.
- The conversation scrolls above a fixed composer.
- The newest exchange is automatically kept in view.
- Sources and connection-status text are not shown in the chat.
- Product names are used by default. HS codes appear only when requested.
- Ordinary answers focus on exports, products, sectors, countries, years, rankings and shares.
- EXPY, HHI, RCA, unrealized potential, performance status, priority and CAGR are shown only when explicitly requested.
- User-facing responses use normal hyphens instead of em dashes.
