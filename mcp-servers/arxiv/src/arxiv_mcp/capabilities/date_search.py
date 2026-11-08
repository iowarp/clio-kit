"""
ArXiv date-based search capabilities.
"""

from datetime import datetime
from typing import Dict, Any
import logging
from .arxiv_base import execute_arxiv_query, field_query

logger = logging.getLogger(__name__)


async def search_date_range(
    start_date: str, end_date: str, category: str = "", max_results: int = 20
) -> Dict[str, Any]:
    """
    Search ArXiv papers within a specific date range.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        category: Optional category filter (e.g., 'cs.AI')
        max_results: Maximum number of results to return (default: 20)

    Returns:
        Dictionary containing search results and metadata
    """
    for value in (start_date, end_date):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date '{value}': expected YYYY-MM-DD") from None

    # Construct date range search query
    date_query = (
        f"submittedDate:[{start_date.replace('-', '')} TO {end_date.replace('-', '')}]"
    )
    if category:
        search_query = f"{field_query('cat', category)} AND {date_query}"
    else:
        search_query = date_query

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    papers = await execute_arxiv_query(params)

    return {
        "success": True,
        "papers": papers,
        "start_date": start_date,
        "end_date": end_date,
        "category": category or "all",
        "max_results": max_results,
        "returned_results": len(papers),
        "message": f"Successfully found {len(papers)} papers from {start_date} to {end_date}",
    }
