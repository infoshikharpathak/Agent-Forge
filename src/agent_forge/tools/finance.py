from __future__ import annotations

"""Finance tools: stock prices and company financials via yfinance."""

from agent_forge.tools import register


@register("stock_price")
def stock_price(ticker: str) -> str:
    """
    Get the current stock price and basic market data for a ticker symbol.
    Example: stock_price("AAPL") or stock_price("MSFT")
    """
    try:
        import yfinance as yf  # type: ignore

        stock = yf.Ticker(ticker.upper())
        info = stock.info

        price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
        change = info.get("regularMarketChangePercent", "N/A")
        market_cap = info.get("marketCap", "N/A")
        if isinstance(market_cap, (int, float)):
            market_cap = f"${market_cap / 1e9:.2f}B"

        return (
            f"Ticker: {ticker.upper()}\n"
            f"Name: {info.get('longName', 'N/A')}\n"
            f"Price: ${price}\n"
            f"Change (today): {change:.2f}%\n"
            f"Market Cap: {market_cap}\n"
            f"52-week High: {info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"52-week Low: {info.get('fiftyTwoWeekLow', 'N/A')}\n"
            f"P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
            f"Sector: {info.get('sector', 'N/A')}"
        )
    except Exception as exc:
        return f"Failed to fetch stock data for '{ticker}': {exc}"


@register("company_financials")
def company_financials(ticker: str) -> str:
    """
    Get key financial metrics for a company: revenue, earnings, margins, and debt.
    Example: company_financials("AAPL")
    """
    try:
        import yfinance as yf  # type: ignore

        stock = yf.Ticker(ticker.upper())
        info = stock.info

        def fmt(val, prefix="$", scale=1e9, suffix="B"):
            if isinstance(val, (int, float)):
                return f"{prefix}{val / scale:.2f}{suffix}"
            return "N/A"

        return (
            f"Ticker: {ticker.upper()} — {info.get('longName', '')}\n"
            f"Revenue (TTM): {fmt(info.get('totalRevenue'))}\n"
            f"Gross Profit: {fmt(info.get('grossProfits'))}\n"
            f"Net Income: {fmt(info.get('netIncomeToCommon'))}\n"
            f"EBITDA: {fmt(info.get('ebitda'))}\n"
            f"Profit Margin: {info.get('profitMargins', 'N/A')}\n"
            f"Operating Margin: {info.get('operatingMargins', 'N/A')}\n"
            f"Total Debt: {fmt(info.get('totalDebt'))}\n"
            f"Total Cash: {fmt(info.get('totalCash'))}\n"
            f"Return on Equity: {info.get('returnOnEquity', 'N/A')}\n"
            f"EPS (TTM): {info.get('trailingEps', 'N/A')}"
        )
    except Exception as exc:
        return f"Failed to fetch financials for '{ticker}': {exc}"
