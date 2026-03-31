"""
Centralized debug logging system for multi-agent pipeline.

Logs:
- Agent-to-agent messages
- Voting and deliberation details
- LLM prompts and responses
- Decision rationales
- Evidence and reasoning chains
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime


class DebugLogger:
    """Centralized logger for detailed debugging of agent decisions and communications."""
    
    def __init__(self, log_file: str = "pipeline_debug.log", verbose: bool = True, clear_log: bool = True):
        self.log_file = Path(log_file)
        self.verbose = verbose
        
        # Only clear if explicitly requested (at start of new run)
        if clear_log:
            self.log_file.write_text("")
        
        self._message_counter = 0
        self._vote_counter = 0
        self._llm_call_counter = 0
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message to both file and terminal."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] [{level}] {message}"
        
        if self.verbose:
            print(formatted)
        
        with open(self.log_file, "a") as f:
            f.write(formatted + "\n")
    
    def log_agent_message(
        self,
        sender: str,
        receiver: str,
        message_type: str,
        content: Dict[str, Any],
        priority: str = "NORMAL"
    ) -> None:
        """Log inter-agent message passing."""
        self._message_counter += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Terminal output (concise)
        content_preview = str(content.get("action", list(content.keys())[:2]))[:50]
        terminal_msg = f"[{timestamp}] [MSG #{self._message_counter}] {sender} → {receiver}: {message_type} ({content_preview}...)"
        
        if self.verbose and priority in ["HIGH", "URGENT"]:
            print(terminal_msg)
        
        # File output (detailed)
        with open(self.log_file, "a") as f:
            f.write(f"\n{'─'*80}\n")
            f.write(f"MESSAGE #{self._message_counter} [{timestamp}]\n")
            f.write(f"From: {sender}\n")
            f.write(f"To: {receiver}\n")
            f.write(f"Type: {message_type}\n")
            f.write(f"Priority: {priority}\n")
            f.write(f"Content:\n{json.dumps(content, indent=2, default=str)}\n")
            f.write(f"{'─'*80}\n")
    
    def log_vote(
        self,
        voter: str,
        hypothesis_id: str,
        vote_type: str,
        confidence: float,
        rationale: str,
        evidence: Optional[List[str]] = None,
        weighted_score: Optional[float] = None
    ) -> None:
        """Log a vote cast by an agent."""
        self._vote_counter += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Terminal output
        symbol = "✓" if "ACCEPT" in vote_type else "✗" if "REJECT" in vote_type else "○"
        score_str = f"{weighted_score:.2f}" if weighted_score is not None else "0.00"
        terminal_msg = f"[{timestamp}] [VOTE #{self._vote_counter}] {voter}: {symbol} {vote_type} (conf={confidence:.2f}, score={score_str}) - {rationale[:60]}..."
        
        if self.verbose:
            print(terminal_msg)
        
        # File output
        with open(self.log_file, "a") as f:
            f.write(f"\n{'═'*80}\n")
            f.write(f"VOTE #{self._vote_counter} [{timestamp}]\n")
            f.write(f"Voter: {voter}\n")
            f.write(f"Hypothesis: {hypothesis_id}\n")
            f.write(f"Vote: {vote_type}\n")
            f.write(f"Confidence: {confidence:.3f}\n")
            if weighted_score is not None:
                f.write(f"Weighted Score: {weighted_score:.3f}\n")
            f.write(f"Rationale: {rationale}\n")
            if evidence:
                f.write(f"Evidence:\n")
                for e in evidence[:3]:
                    f.write(f"  - {e[:100]}...\n")
            f.write(f"{'═'*80}\n")
    
    def log_debate_argument(
        self,
        agent: str,
        hypothesis_id: str,
        position: str,
        argument: str,
        evidence: Optional[List[str]] = None,
        counter_to: Optional[str] = None
    ) -> None:
        """Log a debate argument."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Terminal output
        pos_symbol = "⚡" if position == "support" else "⚠"
        counter_str = f" (countering {counter_to})" if counter_to else ""
        terminal_msg = f"[{timestamp}] [DEBATE] {agent}: {pos_symbol} {position.upper()}{counter_str} - {argument[:60]}..."
        
        if self.verbose:
            print(terminal_msg)
        
        # File output
        with open(self.log_file, "a") as f:
            f.write(f"\n{'▓'*80}\n")
            f.write(f"DEBATE ARGUMENT [{timestamp}]\n")
            f.write(f"Agent: {agent}\n")
            f.write(f"Hypothesis: {hypothesis_id}\n")
            f.write(f"Position: {position}\n")
            if counter_to:
                f.write(f"Countering: {counter_to}\n")
            f.write(f"Argument:\n{argument}\n")
            if evidence:
                f.write(f"Supporting Evidence:\n")
                for e in evidence[:3]:
                    f.write(f"  - {e[:150]}...\n")
            f.write(f"{'▓'*80}\n")
    
    def log_llm_call(
        self,
        agent: str,
        model: str,
        prompt_preview: str,
        response_preview: str,
        tokens_used: Optional[int] = None,
        duration_ms: Optional[float] = None
    ) -> None:
        """Log an LLM API call."""
        self._llm_call_counter += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Terminal output (very concise)
        duration_str = f"{duration_ms:.0f}ms" if duration_ms else "?"
        tokens_str = f"{tokens_used}tok" if tokens_used else "?"
        terminal_msg = f"[{timestamp}] [LLM #{self._llm_call_counter}] {agent} → {model} ({duration_str}, {tokens_str})"
        
        if self.verbose:
            print(terminal_msg)
        
        # File output (full prompt/response)
        with open(self.log_file, "a") as f:
            f.write(f"\n{'▒'*80}\n")
            f.write(f"LLM CALL #{self._llm_call_counter} [{timestamp}]\n")
            f.write(f"Agent: {agent}\n")
            f.write(f"Model: {model}\n")
            if duration_ms:
                f.write(f"Duration: {duration_ms:.2f}ms\n")
            if tokens_used:
                f.write(f"Tokens: {tokens_used}\n")
            f.write(f"\nPrompt (first 500 chars):\n{prompt_preview[:500]}...\n")
            f.write(f"\nResponse (first 500 chars):\n{response_preview[:500]}...\n")
            f.write(f"{'▒'*80}\n")
    
    def log_decision(
        self, 
        agent: str, 
        decision_type: str,
        item: Any,
        decision: str,
        reasoning: str,
        confidence: Optional[float] = None
    ) -> None:
        """Log a specific agent decision with full context."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Terminal output (concise)
        conf_str = f"conf={confidence:.2f}" if confidence is not None else ""
        reasoning = reasoning or ""
        terminal_msg = f"[{timestamp}] [{agent}] {decision_type.upper()}: {decision} ({conf_str}) - {reasoning[:60]}..."
        
        if self.verbose:
            print(terminal_msg)
        
        # File output (detailed with full item)
        with open(self.log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"DECISION [{timestamp}]\n")
            f.write(f"Agent: {agent}\n")
            f.write(f"Type: {decision_type}\n")
            f.write(f"Decision: {decision}\n")
            f.write(f"Reasoning: {reasoning}\n")
            if confidence is not None:
                f.write(f"Confidence: {confidence:.3f}\n")
            f.write(f"\nFull Item:\n")
            f.write(json.dumps(item, indent=2, default=str))
            f.write(f"\n{'='*80}\n")
    
    def log_summary(self, agent: str, summary: Dict[str, Any]) -> None:
        """Log summary statistics for an agent."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        msg = f"[{timestamp}] [{agent}] SUMMARY: {json.dumps(summary, default=str)}"
        
        if self.verbose:
            print(msg)
        
        with open(self.log_file, "a") as f:
            f.write(f"\n{msg}\n")
    
    def log_stage_header(self, stage_num: int, stage_name: str, total_stages: int = 9) -> None:
        """Log a pipeline stage header."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        header = f"\n{'█'*80}\n[{timestamp}] STAGE [{stage_num}/{total_stages}]: {stage_name}\n{'█'*80}"
        
        if self.verbose:
            print(header)
        
        with open(self.log_file, "a") as f:
            f.write(header + "\n")


# Global instance
_logger: Optional[DebugLogger] = None


def get_debug_logger(log_file: str = "pipeline_debug.log") -> DebugLogger:
    """Get or create the global debug logger. Reuses existing instance to preserve logs."""
    global _logger
    if _logger is None:
        _logger = DebugLogger(log_file, clear_log=False)  # Don't clear when accessed later
    return _logger
