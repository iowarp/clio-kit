"""
Comprehensive tests for MCP handlers.
Tests all handler functions and error handling scenarios.
"""

import pytest
import httpx
import asyncio
from unittest.mock import patch, AsyncMock, Mock

from arxiv_mcp import mcp_handlers

from . import settle


@pytest.fixture
def empty_arxiv_api():
    """Keep handler boundary tests independent of the public API rate limit."""
    response = httpx.Response(
        200,
        text='<feed xmlns="http://www.w3.org/2005/Atom"/>',
        request=httpx.Request("GET", "https://export.arxiv.org/api/query"),
    )
    with patch(
        "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=response
    ) as request:
        yield request


class TestMCPHandlers:
    """Test class for MCP handler functions."""

    @pytest.mark.asyncio
    async def test_search_arxiv_handler_success(self):
        """Test search_arxiv_handler with successful response."""
        expected_result = {"papers": [{"id": "test"}]}

        with patch(
            "arxiv_mcp.mcp_handlers.search_arxiv", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = expected_result

            result = await mcp_handlers.search_arxiv_handler("cs.AI", 5)

            assert result == expected_result
            mock_search.assert_called_once_with("cs.AI", 5)

    @pytest.mark.asyncio
    async def test_search_arxiv_handler_error(self):
        """Test search_arxiv_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.search_arxiv", new_callable=AsyncMock
        ) as mock_search:
            mock_search.side_effect = ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                await mcp_handlers.search_arxiv_handler("cs.AI", 5)

    @pytest.mark.asyncio
    async def test_get_recent_papers_handler_success(self):
        """Test get_recent_papers_handler with successful response."""
        expected_result = {"papers": [{"id": "recent"}]}

        with patch(
            "arxiv_mcp.mcp_handlers.get_recent_papers", new_callable=AsyncMock
        ) as mock_recent:
            mock_recent.return_value = expected_result

            result = await mcp_handlers.get_recent_papers_handler("cs.AI", 5)

            assert result == expected_result
            mock_recent.assert_called_once_with("cs.AI", 5)

    @pytest.mark.asyncio
    async def test_get_recent_papers_handler_error(self):
        """Test get_recent_papers_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.get_recent_papers", new_callable=AsyncMock
        ) as mock_recent:
            mock_recent.side_effect = ConnectionError("Network error")

            with pytest.raises(ConnectionError, match="Network error"):
                await mcp_handlers.get_recent_papers_handler("cs.AI", 5)

    @pytest.mark.asyncio
    async def test_search_papers_by_author_handler_success(self):
        """Test search_papers_by_author_handler with successful response."""
        expected_result = {"papers": [{"author": "Test Author"}]}

        with patch(
            "arxiv_mcp.mcp_handlers.search_papers_by_author", new_callable=AsyncMock
        ) as mock_author:
            mock_author.return_value = expected_result

            result = await mcp_handlers.search_papers_by_author_handler(
                "Test Author", 10
            )

            assert result == expected_result
            mock_author.assert_called_once_with("Test Author", 10)

    @pytest.mark.asyncio
    async def test_search_papers_by_author_handler_error(self):
        """Test search_papers_by_author_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.search_papers_by_author", new_callable=AsyncMock
        ) as mock_author:
            mock_author.side_effect = RuntimeError("Runtime error")

            with pytest.raises(RuntimeError, match="Runtime error"):
                await mcp_handlers.search_papers_by_author_handler("Test Author", 10)

    @pytest.mark.asyncio
    async def test_search_by_title_handler_success(self):
        """Test search_by_title_handler with successful response."""
        expected_result = {"papers": [{"title": "Test Title"}]}

        with patch(
            "arxiv_mcp.mcp_handlers.search_by_title", new_callable=AsyncMock
        ) as mock_title:
            mock_title.return_value = expected_result

            result = await mcp_handlers.search_by_title_handler("neural networks", 10)

            assert result == expected_result
            mock_title.assert_called_once_with("neural networks", 10)

    @pytest.mark.asyncio
    async def test_search_by_title_handler_error(self):
        """Test search_by_title_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.search_by_title", new_callable=AsyncMock
        ) as mock_title:
            mock_title.side_effect = Exception("Generic error")

            with pytest.raises(Exception, match="Generic error"):
                await mcp_handlers.search_by_title_handler("neural networks", 10)

    @pytest.mark.asyncio
    async def test_search_by_abstract_handler_success(self):
        """Test search_by_abstract_handler with successful response."""
        expected_result = {"papers": [{"abstract": "machine learning"}]}

        with patch(
            "arxiv_mcp.mcp_handlers.search_by_abstract", new_callable=AsyncMock
        ) as mock_abstract:
            mock_abstract.return_value = expected_result

            result = await mcp_handlers.search_by_abstract_handler(
                "machine learning", 10
            )

            assert result == expected_result
            mock_abstract.assert_called_once_with("machine learning", 10)

    @pytest.mark.asyncio
    async def test_search_by_abstract_handler_error(self):
        """Test search_by_abstract_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.search_by_abstract", new_callable=AsyncMock
        ) as mock_abstract:
            mock_abstract.side_effect = KeyError("Key not found")

            with pytest.raises(KeyError, match="Key not found"):
                await mcp_handlers.search_by_abstract_handler("machine learning", 10)

    @pytest.mark.asyncio
    async def test_search_by_subject_handler_success(self):
        """Test search_by_subject_handler with successful response."""
        expected_result = {"papers": [{"subject": "Computer Science"}]}

        with patch(
            "arxiv_mcp.mcp_handlers.search_by_subject", new_callable=AsyncMock
        ) as mock_subject:
            mock_subject.return_value = expected_result

            result = await mcp_handlers.search_by_subject_handler(
                "Computer Science", 10
            )

            assert result == expected_result
            mock_subject.assert_called_once_with("Computer Science", 10)

    @pytest.mark.asyncio
    async def test_search_by_subject_handler_error(self):
        """Test search_by_subject_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.search_by_subject", new_callable=AsyncMock
        ) as mock_subject:
            mock_subject.side_effect = AttributeError("Attribute error")

            with pytest.raises(AttributeError, match="Attribute error"):
                await mcp_handlers.search_by_subject_handler("Computer Science", 10)

    @pytest.mark.asyncio
    async def test_search_date_range_handler_success(self):
        """Test search_date_range_handler with successful response."""
        expected_result = {"papers": [{"date": "2023-06-15"}]}

        with patch(
            "arxiv_mcp.mcp_handlers.search_date_range", new_callable=AsyncMock
        ) as mock_date:
            mock_date.return_value = expected_result

            result = await mcp_handlers.search_date_range_handler(
                "2023-06-01", "2023-06-30", "", 10
            )

            assert result == expected_result
            mock_date.assert_called_once_with("2023-06-01", "2023-06-30", "", 10)

    @pytest.mark.asyncio
    async def test_search_date_range_handler_error(self):
        """Test search_date_range_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.search_date_range", new_callable=AsyncMock
        ) as mock_date:
            mock_date.side_effect = TypeError("Type error")

            with pytest.raises(TypeError, match="Type error"):
                await mcp_handlers.search_date_range_handler(
                    "2023-06-01", "2023-06-30", "", 10
                )

    @pytest.mark.asyncio
    async def test_get_paper_details_handler_success(self):
        """Test get_paper_details_handler with successful response."""
        expected_result = {"paper": {"id": "2023.12345", "details": True}}

        with patch(
            "arxiv_mcp.mcp_handlers.get_paper_details", new_callable=AsyncMock
        ) as mock_details:
            mock_details.return_value = expected_result

            result = await mcp_handlers.get_paper_details_handler("2023.12345")

            assert result == expected_result
            mock_details.assert_called_once_with("2023.12345")

    @pytest.mark.asyncio
    async def test_get_paper_details_handler_error(self):
        """Test get_paper_details_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.get_paper_details", new_callable=AsyncMock
        ) as mock_details:
            mock_details.side_effect = FileNotFoundError("Paper not found")

            with pytest.raises(FileNotFoundError, match="Paper not found"):
                await mcp_handlers.get_paper_details_handler("2023.12345")

    @pytest.mark.asyncio
    async def test_find_similar_papers_handler_success(self):
        """Test find_similar_papers_handler with successful response."""
        expected_result = {"similar_papers": [{"id": "2023.67890"}]}

        with patch(
            "arxiv_mcp.mcp_handlers.find_similar_papers", new_callable=AsyncMock
        ) as mock_similar:
            mock_similar.return_value = expected_result

            result = await mcp_handlers.find_similar_papers_handler("2023.12345", 5)

            assert result == expected_result
            mock_similar.assert_called_once_with("2023.12345", 5)

    @pytest.mark.asyncio
    async def test_find_similar_papers_handler_error(self):
        """Test find_similar_papers_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.find_similar_papers", new_callable=AsyncMock
        ) as mock_similar:
            mock_similar.side_effect = IndexError("Index error")

            with pytest.raises(IndexError, match="Index error"):
                await mcp_handlers.find_similar_papers_handler("2023.12345", 5)

    @pytest.mark.asyncio
    async def test_export_to_bibtex_handler_success(self):
        """Test export_to_bibtex_handler with successful response."""
        expected_result = {"bibtex": "@article{test,title={Test}}"}

        with patch(
            "arxiv_mcp.mcp_handlers.export_to_bibtex", new_callable=AsyncMock
        ) as mock_export:
            mock_export.return_value = expected_result

            # Pass as JSON string as expected by the handler
            result = await mcp_handlers.export_to_bibtex_handler('["2023.12345"]')

            assert result == expected_result
            mock_export.assert_called_once_with(["2023.12345"])

    @pytest.mark.asyncio
    async def test_export_to_bibtex_handler_error(self):
        """Test export_to_bibtex_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.export_to_bibtex", new_callable=AsyncMock
        ) as mock_export:
            mock_export.side_effect = PermissionError("Permission denied")

            with pytest.raises(PermissionError, match="Permission denied"):
                await mcp_handlers.export_to_bibtex_handler('["2023.12345"]')

    @pytest.mark.asyncio
    async def test_download_paper_pdf_handler_success(self):
        """Test download_paper_pdf_handler with successful response."""
        expected_result = {"status": "success", "file_path": "/tmp/paper.pdf"}

        with patch(
            "arxiv_mcp.mcp_handlers.download_paper_pdf", new_callable=AsyncMock
        ) as mock_download:
            mock_download.return_value = expected_result

            result = await mcp_handlers.download_paper_pdf_handler("2023.12345", "/tmp")

            assert result == expected_result
            mock_download.assert_called_once_with("2023.12345", "/tmp")

    @pytest.mark.asyncio
    async def test_download_paper_pdf_handler_error(self):
        """Test download_paper_pdf_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.download_paper_pdf", new_callable=AsyncMock
        ) as mock_download:
            mock_download.side_effect = OSError("OS error")

            with pytest.raises(OSError, match="OS error"):
                await mcp_handlers.download_paper_pdf_handler("2023.12345", "/tmp")

    @pytest.mark.asyncio
    async def test_get_pdf_url_handler_success(self):
        """Test get_pdf_url_handler with successful response."""
        expected_result = {"pdf_url": "https://arxiv.org/pdf/2023.12345.pdf"}

        with patch(
            "arxiv_mcp.mcp_handlers.get_pdf_url", new_callable=AsyncMock
        ) as mock_url:
            mock_url.return_value = expected_result

            result = await mcp_handlers.get_pdf_url_handler("2023.12345")

            assert result == expected_result
            mock_url.assert_called_once_with("2023.12345")

    @pytest.mark.asyncio
    async def test_get_pdf_url_handler_error(self):
        """Test get_pdf_url_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.get_pdf_url", new_callable=AsyncMock
        ) as mock_url:
            mock_url.side_effect = ImportError("Import error")

            with pytest.raises(ImportError, match="Import error"):
                await mcp_handlers.get_pdf_url_handler("2023.12345")

    @pytest.mark.asyncio
    async def test_download_multiple_pdfs_handler_success(self):
        """Test download_multiple_pdfs_handler with successful response."""
        expected_result = {"status": "success", "downloaded": 2}

        with patch(
            "arxiv_mcp.mcp_handlers.download_multiple_pdfs", new_callable=AsyncMock
        ) as mock_multi:
            mock_multi.return_value = expected_result

            # Pass as JSON string as expected by the handler
            result = await mcp_handlers.download_multiple_pdfs_handler(
                '["2023.12345", "2023.67890"]', "/tmp", 2
            )

            assert result == expected_result
            mock_multi.assert_called_once_with(["2023.12345", "2023.67890"], "/tmp", 2)

    @pytest.mark.asyncio
    async def test_download_multiple_pdfs_handler_error(self):
        """Test download_multiple_pdfs_handler with error handling."""
        with patch(
            "arxiv_mcp.mcp_handlers.download_multiple_pdfs", new_callable=AsyncMock
        ) as mock_multi:
            mock_multi.side_effect = NotImplementedError("Not implemented")

            with pytest.raises(NotImplementedError, match="Not implemented"):
                await mcp_handlers.download_multiple_pdfs_handler(
                    '["2023.12345", "2023.67890"]', "/tmp", 2
                )

    @pytest.mark.asyncio
    async def test_all_handlers_error_format_consistency(self):
        """Test that all handlers return consistent error formats."""

        handlers_to_test = [
            ("search_arxiv_handler", ["cs.AI", 5]),
            ("get_recent_papers_handler", ["cs.AI", 5]),
            ("search_papers_by_author_handler", ["Test Author", 5]),
            ("search_by_title_handler", ["Test Title", 5]),
            ("search_by_abstract_handler", ["Test Abstract", 5]),
            ("search_by_subject_handler", ["Test Subject", 5]),
            ("search_date_range_handler", ["2023-01-01", "2023-12-31", 5]),
            ("get_paper_details_handler", ["test-id"]),
            ("find_similar_papers_handler", ["test-id", 5]),
            ("export_to_bibtex_handler", ['["test-id"]']),
            ("download_paper_pdf_handler", ["test-id", "/tmp"]),
            ("get_pdf_url_handler", ["test-id"]),
            ("download_multiple_pdfs_handler", ['["test-id"]', "/tmp", 1]),
        ]

        for handler_name, args in handlers_to_test:
            if hasattr(mcp_handlers, handler_name):
                handler = getattr(mcp_handlers, handler_name)

                # Mock the underlying function to raise an exception
                func_name = handler_name.replace("_handler", "")
                with patch(
                    f"arxiv_mcp.mcp_handlers.{func_name}", new_callable=AsyncMock
                ) as mock_func:
                    mock_func.side_effect = RuntimeError("Test error")

                    # Failures propagate; no success-shaped error envelope
                    with pytest.raises(RuntimeError, match="Test error"):
                        await handler(*args)

    @pytest.mark.asyncio
    async def test_handlers_with_none_parameters(self, empty_arxiv_api):
        """Test handlers with None parameters."""

        handlers_to_test = [
            ("search_arxiv_handler", [None, 5]),
            ("get_recent_papers_handler", ["cs.AI", None]),
            ("search_papers_by_author_handler", [None, 5]),
            ("get_paper_details_handler", [None]),
        ]

        for handler_name, args in handlers_to_test:
            if hasattr(mcp_handlers, handler_name):
                handler = getattr(mcp_handlers, handler_name)

                # Should handle None parameters gracefully
                result = await settle(handler(*args))
                # May return error or handle gracefully
                assert result is not None

    @pytest.mark.asyncio
    async def test_handlers_with_empty_string_parameters(self):
        """Test handlers with empty string parameters."""

        with patch("httpx.AsyncClient") as mock_client:
            # Mock HTTP responses to speed up the test
            mock_session = Mock()
            mock_client.return_value.__aenter__.return_value = mock_session
            mock_response = Mock()
            mock_response.text.return_value = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""
            mock_response.raise_for_status = Mock()
            mock_session.get.return_value = mock_response

            handlers_to_test = [
                ("search_arxiv_handler", ["", 5]),
                ("search_papers_by_author_handler", ["", 5]),
                ("search_by_title_handler", ["", 5]),
                ("search_by_abstract_handler", ["", 5]),
                ("search_by_subject_handler", ["", 5]),
                ("get_paper_details_handler", [""]),
                ("find_similar_papers_handler", ["", 5]),
            ]

            for handler_name, args in handlers_to_test:
                if hasattr(mcp_handlers, handler_name):
                    handler = getattr(mcp_handlers, handler_name)

                    # Should handle empty strings gracefully
                    result = await settle(handler(*args))
                    assert result is not None

    @pytest.mark.asyncio
    async def test_handlers_with_extreme_parameters(self):
        """Test handlers with extreme parameter values."""

        with patch("httpx.AsyncClient") as mock_client:
            # Mock HTTP responses to speed up the test
            mock_session = Mock()
            mock_client.return_value.__aenter__.return_value = mock_session
            mock_response = Mock()
            mock_response.text.return_value = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""
            mock_response.raise_for_status = Mock()
            mock_session.get.return_value = mock_response

            # Test with very large max_results
            large_result = await settle(
                mcp_handlers.search_arxiv_handler("cs.AI", 10000)
            )
            assert large_result is not None

            # Test with zero max_results
            zero_result = await settle(mcp_handlers.search_arxiv_handler("cs.AI", 0))
            assert zero_result is not None

            # Test with negative max_results
            negative_result = await settle(
                mcp_handlers.search_arxiv_handler("cs.AI", -1)
            )
            assert negative_result is not None

            # Test with very long strings (reduced for speed)
            long_query = "A" * 100  # Reduced from 10000 to 100
            long_result = await settle(
                mcp_handlers.search_by_title_handler(long_query, 5)
            )
            assert long_result is not None

    @pytest.mark.asyncio
    async def test_concurrent_handler_execution(self, empty_arxiv_api):
        """Test concurrent execution of multiple handlers."""

        # Create tasks for concurrent execution
        tasks = [
            settle(mcp_handlers.search_arxiv_handler("cs.AI", 3)),
            settle(mcp_handlers.get_recent_papers_handler("cs.LG", 2)),
            settle(mcp_handlers.search_papers_by_author_handler("Test Author", 1)),
            settle(mcp_handlers.search_by_title_handler("Test Title", 1)),
            settle(mcp_handlers.search_by_abstract_handler("Test Abstract", 1)),
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All tasks should complete
        assert len(results) == len(tasks)

        # None should be exceptions (though they may be error responses)
        for result in results:
            assert not isinstance(result, Exception)

    @pytest.mark.asyncio
    async def test_json_parameter_parsing(self):
        """Test JSON parameter parsing in handlers."""

        with patch("httpx.AsyncClient") as mock_client:
            # Mock HTTP responses to speed up the test
            mock_session = Mock()
            mock_client.return_value.__aenter__.return_value = mock_session
            mock_response = Mock()
            mock_response.text.return_value = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
        <id>http://arxiv.org/abs/2301.12345v1</id>
        <title>Test Paper</title>
        <summary>Test abstract</summary>
        <author><name>Test Author</name></author>
        <published>2023-01-15T00:00:00Z</published>
        <updated>2023-01-15T00:00:00Z</updated>
        <category term="cs.AI" />
    </entry>
</feed>"""
            mock_response.raise_for_status = Mock()
            mock_session.get.return_value = mock_response

            # Test export_to_bibtex_handler with various JSON formats
            json_test_cases = [
                '["2301.12345"]',
                '["2301.12345", "2301.67890"]',
                "[]",
                '["single-id"]',
            ]

            for json_str in json_test_cases:
                result = await settle(mcp_handlers.export_to_bibtex_handler(json_str))
                assert result is not None

            # Test download_multiple_pdfs_handler with mocked file operations
            with patch("os.makedirs"), patch("builtins.open", create=True):
                for json_str in json_test_cases:
                    result = await settle(
                        mcp_handlers.download_multiple_pdfs_handler(json_str, "/tmp", 1)
                    )
                    assert result is not None

    @pytest.mark.asyncio
    async def test_invalid_json_parameter_handling(self):
        """Test handling of invalid JSON parameters."""

        invalid_json_cases = [
            "invalid json",
            "{not a list}",
            "[unclosed",
            "",
            None,
        ]

        for invalid_json in invalid_json_cases:
            # export_to_bibtex_handler should handle invalid JSON
            result = await settle(mcp_handlers.export_to_bibtex_handler(invalid_json))
            # Should return error or handle gracefully
            assert result is not None

            # download_multiple_pdfs_handler should handle invalid JSON
            result = await settle(
                mcp_handlers.download_multiple_pdfs_handler(invalid_json, "/tmp", 1)
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_date_range_edge_cases(self):
        """Test date range handler with edge cases."""

        with patch("httpx.AsyncClient") as mock_client:
            # Mock HTTP responses to speed up the test
            mock_session = Mock()
            mock_client.return_value.__aenter__.return_value = mock_session
            mock_response = Mock()
            mock_response.text.return_value = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""
            mock_response.raise_for_status = Mock()
            mock_session.get.return_value = mock_response

            date_edge_cases = [
                ("2023-01-01", "2023-01-01", 5),  # Same start/end date
                ("2023-12-31", "2023-01-01", 5),  # End before start
                ("invalid-date", "2023-12-31", 5),  # Invalid start date
                ("2023-01-01", "invalid-date", 5),  # Invalid end date
                ("", "", 5),  # Empty dates
                ("2023-02-29", "2023-02-28", 5),  # Invalid leap year date
            ]

            for start_date, end_date, max_results in date_edge_cases:
                result = await settle(
                    mcp_handlers.search_date_range_handler(
                        start_date, end_date, max_results
                    )
                )
                assert result is not None

    @pytest.mark.asyncio
    async def test_paper_id_validation(self):
        """Test paper ID validation across handlers."""

        with patch("httpx.AsyncClient") as mock_client:
            # Mock HTTP responses to speed up the test
            mock_session = Mock()
            mock_client.return_value.__aenter__.return_value = mock_session
            mock_response = Mock()
            mock_response.text.return_value = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""
            mock_response.raise_for_status = Mock()
            mock_session.get.return_value = mock_response

            invalid_paper_ids = [
                "",
                None,
                "invalid.format",
                "toolong.paperid.format.test",
                "special@characters",
                "12345",  # Missing prefix
                "cs/0123456",  # Old format
            ]

            paper_id_handlers = [
                "get_paper_details_handler",
                "find_similar_papers_handler",
                "download_paper_pdf_handler",
                "get_pdf_url_handler",
            ]

            with patch("os.makedirs"), patch("builtins.open", create=True):
                for paper_id in invalid_paper_ids:
                    for handler_name in paper_id_handlers:
                        if hasattr(mcp_handlers, handler_name):
                            handler = getattr(mcp_handlers, handler_name)

                            if handler_name == "download_paper_pdf_handler":
                                result = await settle(handler(paper_id, "/tmp"))
                            elif handler_name == "find_similar_papers_handler":
                                result = await settle(handler(paper_id, 5))
                            else:
                                result = await settle(handler(paper_id))

                            # Should handle invalid IDs gracefully
                            assert result is not None

    @pytest.mark.asyncio
    async def test_handler_performance_characteristics(self):
        """The handler adds negligible overhead on top of the search itself.

        The search is mocked, as everywhere else in this file. Timing an
        unmocked call measured how fast export.arxiv.org happened to be, which
        is why this failed CI at 45s against a 30s bound while asserting
        nothing about our own code.
        """

        import time

        with patch(
            "arxiv_mcp.mcp_handlers.search_arxiv", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = {"success": True, "papers": [{"id": "test"}]}

            start_time = time.time()
            result = await mcp_handlers.search_arxiv_handler("cs.AI", 1)
            end_time = time.time()

        assert (end_time - start_time) < 5
        assert result is not None

    @pytest.mark.asyncio
    async def test_exception_type_coverage(self):
        """Test coverage of different exception types."""

        exception_types = [
            (ValueError, "Value error"),
            (TypeError, "Type error"),
            (RuntimeError, "Runtime error"),
            (ConnectionError, "Connection error"),
            (TimeoutError, "Timeout error"),
            (KeyError, "Key error"),
            (IndexError, "Index error"),
            (AttributeError, "Attribute error"),
            (ImportError, "Import error"),
            (NotImplementedError, "Not implemented"),
        ]

        for exception_type, message in exception_types:
            # Test with search_arxiv_handler as representative
            with patch(
                "arxiv_mcp.mcp_handlers.search_arxiv", new_callable=AsyncMock
            ) as mock_search:
                mock_search.side_effect = exception_type(message)

                with pytest.raises(exception_type, match=message):
                    await mcp_handlers.search_arxiv_handler("cs.AI", 5)
