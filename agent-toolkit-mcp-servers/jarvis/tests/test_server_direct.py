"""
Direct tests for server.py tool functions to achieve >90% coverage.
Tests actual function bodies by patching handlers at the capabilities layer.
"""

import pytest
from unittest.mock import Mock, patch


class TestPipelineToolsDirect:
    """Test pipeline tool implementations directly."""

    @pytest.mark.asyncio
    async def test_update_pipeline_tool_direct(self):
        """Test update_pipeline_tool with mocked handler."""
        with patch("src.server.update_pipeline") as mock_handler:
            mock_handler.return_value = {"pipeline_id": "test", "status": "updated"}

            # Import and call after patching
            from src.server import update_pipeline_tool

            result = await update_pipeline_tool.fn("test")

            assert result["pipeline_id"] == "test"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_pipeline_env_tool_direct(self):
        """Test build_pipeline_env_tool with mocked handler."""
        with patch("src.server.build_pipeline_env") as mock_handler:
            mock_handler.return_value = {
                "pipeline_id": "test",
                "status": "environment_built",
            }

            from src.server import build_pipeline_env_tool

            result = await build_pipeline_env_tool.fn("test")

            assert result["status"] == "environment_built"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_pipeline_tool_direct(self):
        """Test create_pipeline_tool with mocked handler."""
        with patch("src.server.create_pipeline") as mock_handler:
            mock_handler.return_value = {"pipeline_id": "new", "status": "created"}

            from src.server import create_pipeline_tool

            result = await create_pipeline_tool.fn("new")

            assert result["pipeline_id"] == "new"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_pipeline_tool_direct(self):
        """Test load_pipeline_tool with mocked handler."""
        with patch("src.server.load_pipeline") as mock_handler:
            mock_handler.return_value = {"pipeline_id": "loaded", "status": "loaded"}

            from src.server import load_pipeline_tool

            result = await load_pipeline_tool.fn("loaded")

            assert result["status"] == "loaded"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_pipeline_tool_no_id_direct(self):
        """Test load_pipeline_tool without ID."""
        with patch("src.server.load_pipeline") as mock_handler:
            mock_handler.return_value = {"pipeline_id": None, "status": "loaded"}

            from src.server import load_pipeline_tool

            result = await load_pipeline_tool.fn(None)

            assert result["status"] == "loaded"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_pkg_config_tool_direct(self):
        """Test get_pkg_config_tool with mocked handler."""
        with patch("src.server.get_pkg_config") as mock_handler:
            mock_handler.return_value = {
                "pipeline_id": "test",
                "pkg_id": "pkg1",
                "config": {},
            }

            from src.server import get_pkg_config_tool

            result = await get_pkg_config_tool.fn("test", "pkg1")

            assert result["pkg_id"] == "pkg1"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_append_pkg_tool_direct(self):
        """Test append_pkg_tool with mocked handler."""
        with patch("src.server.append_pkg") as mock_handler:
            mock_handler.return_value = {"pipeline_id": "test", "appended": "pkg"}

            from src.server import append_pkg_tool

            result = await append_pkg_tool.fn("test", "pkg_type")

            assert result["appended"] == "pkg"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_append_pkg_tool_with_args_direct(self):
        """Test append_pkg_tool with optional args."""
        with patch("src.server.append_pkg") as mock_handler:
            mock_handler.return_value = {"pipeline_id": "test", "appended": "pkg"}

            from src.server import append_pkg_tool

            result = await append_pkg_tool.fn(
                "test",
                "pkg_type",
                pkg_id="pkg1",
                do_configure=False,
                extra_args={"key": "val"},
            )

            assert result["appended"] == "pkg"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_configure_pkg_tool_direct(self):
        """Test configure_pkg_tool with mocked handler."""
        with patch("src.server.configure_pkg") as mock_handler:
            mock_handler.return_value = {"pipeline_id": "test", "configured": "pkg"}

            from src.server import configure_pkg_tool

            result = await configure_pkg_tool.fn("test", "pkg")

            assert result["configured"] == "pkg"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_unlink_pkg_tool_direct(self):
        """Test unlink_pkg_tool with mocked handler."""
        with patch("src.server.unlink_pkg") as mock_handler:
            mock_handler.return_value = {"pipeline_id": "test", "unlinked": "pkg"}

            from src.server import unlink_pkg_tool

            result = await unlink_pkg_tool.fn("test", "pkg")

            assert result["unlinked"] == "pkg"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_pkg_tool_direct(self):
        """Test remove_pkg_tool with mocked handler."""
        with patch("src.server.remove_pkg") as mock_handler:
            mock_handler.return_value = {"pipeline_id": "test", "removed": "pkg"}

            from src.server import remove_pkg_tool

            result = await remove_pkg_tool.fn("test", "pkg")

            assert result["removed"] == "pkg"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_pipeline_tool_direct(self):
        """Test run_pipeline_tool with mocked handler."""
        with patch("src.server.run_pipeline") as mock_handler:
            mock_handler.return_value = {"pipeline_id": "test", "status": "running"}

            from src.server import run_pipeline_tool

            result = await run_pipeline_tool.fn("test")

            assert result["status"] == "running"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_destroy_pipeline_tool_direct(self):
        """Test destroy_pipeline_tool with mocked handler."""
        with patch("src.server.destroy_pipeline") as mock_handler:
            mock_handler.return_value = {"pipeline_id": "test", "status": "destroyed"}

            from src.server import destroy_pipeline_tool

            result = await destroy_pipeline_tool.fn("test")

            assert result["status"] == "destroyed"
            mock_handler.assert_called_once()


class TestJarvisManagerToolsDirect:
    """Test JarvisManager tool implementations directly."""

    def test_jm_create_config_direct(self):
        """Test jm_create_config with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.create.return_value = None
            mock_mgr.save.return_value = None

            from src.server import jm_create_config

            result = jm_create_config.fn("/cfg", "/priv", "/share")

            assert len(result) == 1
            assert "initialized" in result[0]["text"].lower()
            mock_mgr.create.assert_called_once()
            mock_mgr.save.assert_called_once()

    def test_jm_create_config_error_direct(self):
        """Test jm_create_config error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.create.side_effect = Exception("Error")

            from src.server import jm_create_config

            result = jm_create_config.fn("/cfg", "/priv")

            assert "Error" in result[0]["text"]

    def test_jm_load_config_direct(self):
        """Test jm_load_config with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.load.return_value = None

            from src.server import jm_load_config

            result = jm_load_config.fn()

            assert len(result) == 1
            assert "loaded" in result[0]["text"].lower()
            mock_mgr.load.assert_called_once()

    def test_jm_load_config_error_direct(self):
        """Test jm_load_config error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.load.side_effect = Exception("Load error")

            from src.server import jm_load_config

            result = jm_load_config.fn()

            assert "Error" in result[0]["text"]

    def test_jm_save_config_direct(self):
        """Test jm_save_config with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.save.return_value = None

            from src.server import jm_save_config

            result = jm_save_config.fn()

            assert "saved" in result[0]["text"].lower()
            mock_mgr.save.assert_called_once()

    def test_jm_save_config_error_direct(self):
        """Test jm_save_config error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.save.side_effect = Exception("Save error")

            from src.server import jm_save_config

            result = jm_save_config.fn()

            assert "Error" in result[0]["text"]

    def test_jm_set_hostfile_direct(self):
        """Test jm_set_hostfile with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.set_hostfile.return_value = None
            mock_mgr.save.return_value = None

            from src.server import jm_set_hostfile

            result = jm_set_hostfile.fn("/path/host")

            assert "/path/host" in result[0]["text"]
            mock_mgr.set_hostfile.assert_called_once()
            mock_mgr.save.assert_called_once()

    def test_jm_set_hostfile_error_direct(self):
        """Test jm_set_hostfile error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.set_hostfile.side_effect = Exception("Hostfile error")

            from src.server import jm_set_hostfile

            result = jm_set_hostfile.fn("/path")

            assert "Error" in result[0]["text"]

    def test_jm_bootstrap_from_direct(self):
        """Test jm_bootstrap_from with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.bootstrap_from.return_value = None

            from src.server import jm_bootstrap_from

            result = jm_bootstrap_from.fn("machine1")

            assert "machine1" in result[0]["text"].lower()
            mock_mgr.bootstrap_from.assert_called_once()

    def test_jm_bootstrap_from_error_direct(self):
        """Test jm_bootstrap_from error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.bootstrap_from.side_effect = Exception("Bootstrap error")

            from src.server import jm_bootstrap_from

            result = jm_bootstrap_from.fn("machine")

            assert "Error" in result[0]["text"]

    def test_jm_bootstrap_list_direct(self):
        """Test jm_bootstrap_list with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.bootstrap_list.return_value = ["m1", "m2"]

            from src.server import jm_bootstrap_list

            result = jm_bootstrap_list.fn()

            assert len(result) == 2
            assert result[0]["text"] == "m1"
            mock_mgr.bootstrap_list.assert_called_once()

    def test_jm_bootstrap_list_error_direct(self):
        """Test jm_bootstrap_list error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.bootstrap_list.side_effect = Exception("List error")

            from src.server import jm_bootstrap_list

            result = jm_bootstrap_list.fn()

            assert "Error" in result[0]["text"]

    def test_jm_reset_direct(self):
        """Test jm_reset with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.reset.return_value = None

            from src.server import jm_reset

            result = jm_reset.fn()

            assert "reset" in result[0]["text"].lower()
            mock_mgr.reset.assert_called_once()

    def test_jm_reset_error_direct(self):
        """Test jm_reset error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.reset.side_effect = Exception("Reset error")

            from src.server import jm_reset

            result = jm_reset.fn()

            assert "Error" in result[0]["text"]

    def test_jm_list_pipelines_direct(self):
        """Test jm_list_pipelines with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.list_pipelines.return_value = ["p1", "p2"]

            from src.server import jm_list_pipelines

            result = jm_list_pipelines.fn()

            assert len(result) == 2
            mock_mgr.list_pipelines.assert_called_once()

    def test_jm_list_pipelines_error_direct(self):
        """Test jm_list_pipelines error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.list_pipelines.side_effect = Exception("List error")

            from src.server import jm_list_pipelines

            result = jm_list_pipelines.fn()

            assert "Error" in result[0]["text"]

    def test_jm_cd_direct(self):
        """Test jm_cd with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.cd.return_value = None
            mock_mgr.save.return_value = None

            from src.server import jm_cd

            result = jm_cd.fn("pipe1")

            assert "pipe1" in result[0]["text"]
            mock_mgr.cd.assert_called_once()

    def test_jm_cd_error_direct(self):
        """Test jm_cd error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.cd.side_effect = Exception("CD error")

            from src.server import jm_cd

            result = jm_cd.fn("pipe")

            assert "Error" in result[0]["text"]

    def test_jm_list_repos_direct(self):
        """Test jm_list_repos with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.list_repos.return_value = ["repo1", "repo2"]

            from src.server import jm_list_repos

            result = jm_list_repos.fn()

            assert len(result) == 2
            mock_mgr.list_repos.assert_called_once()

    def test_jm_list_repos_error_direct(self):
        """Test jm_list_repos error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.list_repos.side_effect = Exception("List error")

            from src.server import jm_list_repos

            result = jm_list_repos.fn()

            assert "Error" in result[0]["text"]

    def test_jm_add_repo_direct(self):
        """Test jm_add_repo with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.add_repo.return_value = None
            mock_mgr.save.return_value = None

            from src.server import jm_add_repo

            result = jm_add_repo.fn("/repo", True)

            assert "/repo" in result[0]["text"]
            mock_mgr.add_repo.assert_called_once()

    def test_jm_add_repo_error_direct(self):
        """Test jm_add_repo error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.add_repo.side_effect = Exception("Add error")

            from src.server import jm_add_repo

            result = jm_add_repo.fn("/repo")

            assert "Error" in result[0]["text"]

    def test_jm_remove_repo_direct(self):
        """Test jm_remove_repo with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.remove_repo.return_value = None
            mock_mgr.save.return_value = None

            from src.server import jm_remove_repo

            result = jm_remove_repo.fn("repo1")

            assert "repo1" in result[0]["text"]
            mock_mgr.remove_repo.assert_called_once()

    def test_jm_remove_repo_error_direct(self):
        """Test jm_remove_repo error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.remove_repo.side_effect = Exception("Remove error")

            from src.server import jm_remove_repo

            result = jm_remove_repo.fn("repo")

            assert "Error" in result[0]["text"]

    def test_jm_promote_repo_direct(self):
        """Test jm_promote_repo with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.promote_repo.return_value = None
            mock_mgr.save.return_value = None

            from src.server import jm_promote_repo

            result = jm_promote_repo.fn("repo1")

            assert "repo1" in result[0]["text"]
            mock_mgr.promote_repo.assert_called_once()

    def test_jm_promote_repo_error_direct(self):
        """Test jm_promote_repo error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.promote_repo.side_effect = Exception("Promote error")

            from src.server import jm_promote_repo

            result = jm_promote_repo.fn("repo")

            assert "Error" in result[0]["text"]

    def test_jm_get_repo_direct(self):
        """Test jm_get_repo with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_repo = Mock()
            mock_repo.__str__ = Mock(return_value="RepoInfo")
            mock_mgr.get_repo.return_value = mock_repo

            from src.server import jm_get_repo

            result = jm_get_repo.fn("repo1")

            assert "RepoInfo" in result[0]["text"]
            mock_mgr.get_repo.assert_called_once()

    def test_jm_get_repo_error_direct(self):
        """Test jm_get_repo error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.get_repo.side_effect = Exception("Get error")

            from src.server import jm_get_repo

            result = jm_get_repo.fn("repo")

            assert "Error" in result[0]["text"]

    def test_jm_construct_pkg_direct(self):
        """Test jm_construct_pkg with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_pkg = Mock()
            mock_pkg.__class__.__name__ = "TestPkg"
            mock_mgr.construct_pkg.return_value = mock_pkg

            from src.server import jm_construct_pkg

            result = jm_construct_pkg.fn("test_type")

            assert "TestPkg" in result[0]["text"]
            mock_mgr.construct_pkg.assert_called_once()

    def test_jm_construct_pkg_error_direct(self):
        """Test jm_construct_pkg error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.construct_pkg.side_effect = Exception("Construct error")

            from src.server import jm_construct_pkg

            result = jm_construct_pkg.fn("type")

            assert "Error" in result[0]["text"]

    def test_jm_graph_show_direct(self):
        """Test jm_graph_show with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.resource_graph_show.return_value = None

            from src.server import jm_graph_show

            result = jm_graph_show.fn()

            assert "Resource graph" in result[0]["text"]
            mock_mgr.resource_graph_show.assert_called_once()

    def test_jm_graph_show_error_direct(self):
        """Test jm_graph_show error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.resource_graph_show.side_effect = Exception("Show error")

            from src.server import jm_graph_show

            result = jm_graph_show.fn()

            assert "Error" in result[0]["text"]

    def test_jm_graph_build_direct(self):
        """Test jm_graph_build with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.resource_graph_build.return_value = None

            from src.server import jm_graph_build

            result = jm_graph_build.fn(0.5)

            assert "built" in result[0]["text"].lower()
            mock_mgr.resource_graph_build.assert_called_once()

    def test_jm_graph_build_error_direct(self):
        """Test jm_graph_build error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.resource_graph_build.side_effect = Exception("Build error")

            from src.server import jm_graph_build

            result = jm_graph_build.fn(1.0)

            assert "Error" in result[0]["text"]

    def test_jm_graph_modify_direct(self):
        """Test jm_graph_modify with mocked manager."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.resource_graph_modify.return_value = None

            from src.server import jm_graph_modify

            result = jm_graph_modify.fn(0.75)

            assert "modified" in result[0]["text"].lower()
            mock_mgr.resource_graph_modify.assert_called_once()

    def test_jm_graph_modify_error_direct(self):
        """Test jm_graph_modify error handling."""
        with patch("src.server.manager") as mock_mgr:
            mock_mgr.resource_graph_modify.side_effect = Exception("Modify error")

            from src.server import jm_graph_modify

            result = jm_graph_modify.fn(1.0)

            assert "Error" in result[0]["text"]


class TestMainFunctionDirect:
    """Test the main() function entry point."""

    def test_main_stdio_default(self):
        """Test main() with stdio transport."""
        with patch("src.server.os.getenv", return_value="stdio"):
            with patch("src.server.mcp.run") as mock_run:
                with patch("src.server.print"):
                    mock_run.side_effect = [
                        None,
                        KeyboardInterrupt(),
                    ]  # Exit after first call

                    from src.server import main

                    try:
                        main()
                    except KeyboardInterrupt:
                        pass

                    # Should be called at least once
                    assert mock_run.call_count >= 1

    def test_main_sse_transport(self):
        """Test main() with SSE transport."""

        def mock_getenv(key, default=None):
            values = {
                "MCP_TRANSPORT": "sse",
                "MCP_SSE_HOST": "localhost",
                "MCP_SSE_PORT": "9000",
            }
            return values.get(key, default)

        with patch("src.server.os.getenv", side_effect=mock_getenv):
            with patch("src.server.mcp.run") as mock_run:
                with patch("src.server.print"):
                    mock_run.side_effect = KeyboardInterrupt()

                    from src.server import main

                    try:
                        main()
                    except KeyboardInterrupt:
                        pass

                    # Verify SSE was attempted
                    assert mock_run.call_count >= 1
                    call_kwargs = mock_run.call_args[1]
                    assert call_kwargs["transport"] == "sse"
                    assert call_kwargs["host"] == "localhost"
                    assert call_kwargs["port"] == 9000
