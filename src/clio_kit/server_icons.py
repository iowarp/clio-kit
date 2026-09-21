"""Server icons shared by the reference and marketplace generators."""

SERVER_ICONS = {
    "adios": "📊",
    "arxiv": "📄",
    "hdf5": "🗂️",
    "pandas": "🐼",
    "parquet": "📋",
    "plot": "📈",
    "darshan": "⚡",
    "slurm": "🖥️",
    "lmod": "📦",
    "node_hardware": "💻",
    "compression": "🗜️",
    "parallel_sort": "🔄",
    "jarvis": "🤖",
    "chronolog": "⏰",
}


def server_icon(name: str) -> str:
    return SERVER_ICONS.get(name.replace("-", "_"), "🔧")
