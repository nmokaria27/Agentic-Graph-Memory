"""
Patches MemoryAgentBench's agent.py to add "agent_graph_memory" support.

Run once from inside the cloned MemoryAgentBench repo:
    python /path/to/Agent-Graph-Memory/evals/MemoryAgentBench/patch_memagentbench.py

This adds a new branch in AgentWrapper._initialize_agent_by_type() and
AgentWrapper.send_message() so the official harness (main.py) can use
Agent-Graph-Memory as a drop-in agent.
"""

from __future__ import annotations

import sys
from pathlib import Path


IMPORT_INJECTION = '''
# --- Agent-Graph-Memory integration ---
try:
    import sys as _sys
    _agm_root = "{agm_root}"
    if _agm_root not in _sys.path:
        _sys.path.insert(0, _agm_root)
    from evals.MemoryAgentBench.agent_graph_memory_adapter import AgentGraphMemoryWrapper as _AGMWrapper
    _AGM_AVAILABLE = True
except ImportError as _e:
    _AGM_AVAILABLE = False
    print(f"WARNING: Agent-Graph-Memory adapter not found: {{_e}}")
# --- end Agent-Graph-Memory integration ---
'''

INIT_INJECTION = '''
        elif self.agent_name.startswith("agent_graph_memory"):
            if not _AGM_AVAILABLE:
                raise ImportError("Agent-Graph-Memory not found. Set correct path in patch_memagentbench.py")
            self.agent = _AGMWrapper(
                model=agent_config.get("model", "gemma4:31b"),
                embedding_model=agent_config.get("embedding_model", "mxbai-embed-large"),
                retrieval_mode=agent_config.get("retrieval_mode", "hybrid"),
                use_advanced_qa=agent_config.get("use_advanced_qa", True),
                governance_mode=agent_config.get("governance_mode", "permissive"),
                temperature=agent_config.get("temperature", 0.0),
            )
'''

SEND_MSG_INJECTION = '''
        elif self.agent_name.startswith("agent_graph_memory"):
            result = self.agent.send_message(
                message, memorizing=memorizing, query_id=query_id, context_id=context_id
            )
            if memorizing:
                return "Memorized"
            return result
'''


def patch(agent_py_path: Path, agm_root: str) -> None:
    src = agent_py_path.read_text(encoding="utf-8")

    if "agent_graph_memory" in src:
        print("agent.py already patched.")
        return

    # 1. Inject import after last import line
    import_block = IMPORT_INJECTION.format(agm_root=agm_root)
    last_import_pos = 0
    for i, line in enumerate(src.splitlines(keepends=True)):
        if line.startswith("import ") or line.startswith("from "):
            last_import_pos = sum(len(l) for l in src.splitlines(keepends=True)[:i+1])
    src = src[:last_import_pos] + import_block + src[last_import_pos:]

    # 2. Inject into _initialize_agent_by_type — find last elif block
    marker = "def send_message("
    init_insert_pos = src.rfind("elif self.agent_name", 0, src.find(marker))
    block_end = src.find("\n\n", init_insert_pos)
    src = src[:block_end] + INIT_INJECTION + src[block_end:]

    # 3. Inject into send_message — find last elif block
    send_start = src.find(marker)
    last_elif = src.rfind("elif self.agent_name", send_start)
    block_end2 = src.find("\n\n", last_elif)
    if block_end2 == -1:
        block_end2 = len(src)
    src = src[:block_end2] + SEND_MSG_INJECTION + src[block_end2:]

    agent_py_path.write_text(src, encoding="utf-8")
    print(f"Patched {agent_py_path}")


def main() -> None:
    if len(sys.argv) < 2:
        # Try to detect MemoryAgentBench location
        candidates = [
            Path.cwd() / "agent.py",
            Path.cwd().parent / "MemoryAgentBench" / "agent.py",
        ]
        agent_py = next((p for p in candidates if p.exists()), None)
        if agent_py is None:
            print("Usage: python patch_memagentbench.py <path/to/MemoryAgentBench>")
            sys.exit(1)
    else:
        bench_dir = Path(sys.argv[1])
        agent_py = bench_dir / "agent.py"
        if not agent_py.exists():
            print(f"agent.py not found in {bench_dir}")
            sys.exit(1)

    agm_root = str(Path(__file__).parent.parent.parent.resolve())
    print(f"Agent-Graph-Memory root: {agm_root}")
    print(f"Patching: {agent_py}")
    patch(agent_py, agm_root)


if __name__ == "__main__":
    main()
