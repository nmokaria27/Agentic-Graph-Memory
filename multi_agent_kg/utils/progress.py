"""Rich-based interactive progress display for the KG extraction pipeline.

Provides clean, professional terminal output for each stage of the multi-agent
knowledge graph extraction pipeline. Falls back gracefully to plain print()
if the rich library is not installed.

Usage:
    progress = PipelineProgress(total_stages=8)
    progress.print_header(config)
    progress.stage_start(1, "Document Processing", "Segmenting input text")
    progress.stage_detail("Splitting into 13 segments with 10% overlap")
    progress.stage_complete(1, {"segments": 13})

    with progress.batch_progress(13, "Processing segments") as advance:
        for seg in segments:
            process(seg)
            advance()

    progress.print_summary(results)
"""

import time
from typing import Any, Dict, Optional
from contextlib import contextmanager

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
        TaskID,
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Stage names used in the 10-stage deliberative pipeline.
PIPELINE_STAGES = [
    "Adaptive Planning",
    "Document Processing",
    "Domain Classification",
    "Fast First Pass (GLiNER/GLiREL)",
    "Entity Extraction",
    "Entity Resolution",
    "Relation Extraction",
    "Triplex + Schema Alignment",
    "Critic-Corrector Verification",
    "Knowledge Graph Integration",
]


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


class PipelineProgress:
    """Interactive progress display for the KG extraction pipeline.

    Falls back gracefully to plain print() if rich is not installed.
    """

    def __init__(self, total_stages: int = 8):
        self.total_stages = total_stages
        self._stage_start_time: Optional[float] = None
        self._current_stage: Optional[int] = None
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

    # ------------------------------------------------------------------
    # Pipeline header
    # ------------------------------------------------------------------

    def print_header(self, config: Dict[str, Any]) -> None:
        """Print the pipeline configuration header.

        Expected *config* keys (all optional):
            model_tiers   - dict mapping tier name to model name
            agent_models  - dict mapping agent name to model name
            features      - dict of feature_name -> bool
            quality       - dict with 'threshold', 'max_iterations'
            deliberation  - dict with 'voting_agents', 'consensus_threshold', 'min_votes'
        """
        if RICH_AVAILABLE:
            self._print_header_rich(config)
        else:
            self._print_header_plain(config)

    def _print_header_rich(self, config: Dict[str, Any]) -> None:
        """Render a Rich panel with pipeline configuration."""
        lines = Text()

        # Model tiers
        model_tiers = config.get("model_tiers", {})
        if model_tiers:
            lines.append("Model Tiers\n", style="bold underline")
            for tier, model in model_tiers.items():
                tier_label = tier.value if hasattr(tier, "value") else str(tier)
                lines.append(f"  {tier_label:<8}", style="cyan")
                lines.append(f" {model}\n")

        # Per-agent model overrides
        agent_models = config.get("agent_models", {})
        if agent_models:
            lines.append("\nPer-Agent Models\n", style="bold underline")
            for agent_name, model in sorted(agent_models.items()):
                lines.append(f"  {agent_name:<30}", style="cyan")
                lines.append(f" {model}\n", style="dim")

        # Features
        features = config.get("features", {})
        if features:
            lines.append("\nFeatures\n", style="bold underline")
            for name, enabled in features.items():
                marker_style = "green" if enabled else "red"
                marker = "ON " if enabled else "OFF"
                label = name.replace("_", " ").title()
                lines.append(f"  {label:<30} ", style="white")
                lines.append(f"{marker}\n", style=marker_style)

        # Quality settings
        quality = config.get("quality", {})
        if quality:
            lines.append("\nQuality Settings\n", style="bold underline")
            threshold = quality.get("threshold")
            if threshold is not None:
                lines.append(f"  Acceptance Threshold        ", style="white")
                lines.append(f"{threshold}\n", style="green")
            max_iter = quality.get("max_iterations")
            if max_iter is not None:
                lines.append(f"  Max Refinement Iterations   ", style="white")
                lines.append(f"{max_iter}\n", style="green")

        # Deliberation settings
        delib = config.get("deliberation", {})
        if delib:
            lines.append("\nDeliberation\n", style="bold underline")
            agents = delib.get("voting_agents")
            if agents:
                agents_str = ", ".join(agents) if isinstance(agents, list) else str(agents)
                lines.append(f"  Voting Agents               ", style="white")
                lines.append(f"{agents_str}\n", style="dim")
            consensus = delib.get("consensus_threshold")
            if consensus is not None:
                lines.append(f"  Consensus Threshold         ", style="white")
                lines.append(f"{consensus}\n", style="green")
            min_votes = delib.get("min_votes")
            if min_votes is not None:
                lines.append(f"  Min Votes Required          ", style="white")
                lines.append(f"{min_votes}\n", style="green")

        panel = Panel(
            lines,
            title="[bold]Deliberative Multi-Agent KG Pipeline[/bold]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)

    def _print_header_plain(self, config: Dict[str, Any]) -> None:
        """Render a plain-text pipeline configuration header."""
        print()
        print("=" * 70)
        print("DELIBERATIVE MULTI-AGENT KG PIPELINE")
        print("=" * 70)

        model_tiers = config.get("model_tiers", {})
        if model_tiers:
            print("Model Tiers:")
            for tier, model in model_tiers.items():
                tier_label = tier.value if hasattr(tier, "value") else str(tier)
                print(f"  {tier_label}: {model}")

        agent_models = config.get("agent_models", {})
        if agent_models:
            print("\nPer-Agent Models:")
            for agent_name, model in sorted(agent_models.items()):
                print(f"  {agent_name}: {model}")

        features = config.get("features", {})
        if features:
            print("\nFeatures:")
            for name, enabled in features.items():
                status = "Enabled" if enabled else "Disabled"
                label = name.replace("_", " ").title()
                print(f"  {label}: {status}")

        quality = config.get("quality", {})
        if quality:
            print("\nQuality Settings:")
            threshold = quality.get("threshold")
            if threshold is not None:
                print(f"  Threshold: {threshold}")
            max_iter = quality.get("max_iterations")
            if max_iter is not None:
                print(f"  Max Refinement Iterations: {max_iter}")

        delib = config.get("deliberation", {})
        if delib:
            print("\nDeliberation:")
            agents = delib.get("voting_agents")
            if agents:
                agents_str = ", ".join(agents) if isinstance(agents, list) else str(agents)
                print(f"  Voting Agents: {agents_str}")
            consensus = delib.get("consensus_threshold")
            if consensus is not None:
                print(f"  Consensus Threshold: {consensus}")
            min_votes = delib.get("min_votes")
            if min_votes is not None:
                print(f"  Min Votes Required: {min_votes}")

        print("=" * 70)
        print()

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------

    def stage_start(self, stage_num: int, name: str, description: str = "") -> None:
        """Mark the start of a pipeline stage.

        Prints a line like:
            [1/8] Document Processing -- Segmenting input text
        """
        self._stage_start_time = time.time()
        self._current_stage = stage_num

        if RICH_AVAILABLE:
            tag = Text(f" [{stage_num}/{self.total_stages}] ", style="bold cyan")
            stage_name = Text(name, style="bold white")
            if description:
                desc = Text(f" -- {description}", style="dim")
            else:
                desc = Text("")
            line = Text()
            line.append_text(tag)
            line.append_text(stage_name)
            line.append_text(desc)
            self.console.print()
            self.console.print(line)
        else:
            desc_part = f" -- {description}" if description else ""
            print(f"\n[{stage_num}/{self.total_stages}] {name}{desc_part}")
            print("-" * 50)

    def stage_detail(self, message: str) -> None:
        """Print a detail line indented under the current stage."""
        if RICH_AVAILABLE:
            line = Text()
            line.append("     ", style="")
            line.append("|- ", style="dim")
            line.append(message)
            self.console.print(line)
        else:
            print(f"  |- {message}")

    def stage_complete(self, stage_num: int, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Mark stage completion with elapsed time and key metrics.

        *metrics* is a flat dict of label -> value, e.g.
            {"segments": 13, "confidence": 0.92}
        """
        elapsed = 0.0
        if self._stage_start_time is not None:
            elapsed = time.time() - self._stage_start_time
            self._stage_start_time = None

        metrics = metrics or {}
        metrics_parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_parts.append(f"{key}: {value:.2f}")
            else:
                metrics_parts.append(f"{key}: {value}")
        metrics_str = ", ".join(metrics_parts)

        if RICH_AVAILABLE:
            line = Text()
            line.append("     ", style="")
            line.append("\\- ", style="dim")
            line.append("Complete ", style="bold green")
            line.append(f"({_format_elapsed(elapsed)})", style="dim")
            if metrics_str:
                line.append(f" -- {metrics_str}", style="green")
            self.console.print(line)
        else:
            suffix = f" -- {metrics_str}" if metrics_str else ""
            print(f"  \\- Complete ({_format_elapsed(elapsed)}){suffix}")

    # ------------------------------------------------------------------
    # Batch progress (context manager)
    # ------------------------------------------------------------------

    @contextmanager
    def batch_progress(self, total: int, description: str = "Processing"):
        """Return a context manager for batch progress.

        Usage:
            with progress.batch_progress(13, "Processing segments") as advance:
                for seg in segments:
                    process(seg)
                    advance()          # or advance(1, "Segment 3/13")
        """
        if RICH_AVAILABLE and total > 1:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task(description, total=total)

                def advance(n: int = 1, msg: Optional[str] = None) -> None:
                    progress.update(task, advance=n, description=msg or description)

                yield advance
        else:
            # Fallback: simple counter printed inline.
            state = {"count": 0}

            def advance(n: int = 1, msg: Optional[str] = None) -> None:
                state["count"] += n
                label = msg or description
                print(f"    {label}: {state['count']}/{total}")

            yield advance

    # ------------------------------------------------------------------
    # Document header (for corpus processing)
    # ------------------------------------------------------------------

    def print_document_header(
        self,
        doc_id: str,
        doc_num: int = 1,
        total_docs: int = 1,
    ) -> None:
        """Print document processing header."""
        if RICH_AVAILABLE:
            title_text = Text()
            title_text.append(f"Document {doc_num}/{total_docs}", style="bold cyan")
            title_text.append(f"  {doc_id}", style="white")
            panel = Panel(
                title_text,
                border_style="blue",
                box=box.HEAVY,
                expand=True,
                padding=(0, 1),
            )
            self.console.print()
            self.console.print(panel)
        else:
            print()
            print("=" * 70)
            print(f"Document {doc_num}/{total_docs}: {doc_id}")
            print("=" * 70)

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print the final results summary as a Rich table.

        Expected *results* keys (all optional, missing keys show as dashes):
            entities_extracted, entities_resolved, entities_merged,
            triples_extracted, approved_triples, rejected_triples,
            kg_entities, kg_triples,
            processing_time_seconds, relation_library_size,
            fast_entities, fast_triples, critic_iterations,
            domain, segments
        """
        if RICH_AVAILABLE:
            self._print_summary_rich(results)
        else:
            self._print_summary_plain(results)

    def _print_summary_rich(self, results: Dict[str, Any]) -> None:
        """Render the summary as a Rich table."""
        table = Table(
            title="Pipeline Results",
            box=box.ROUNDED,
            title_style="bold white",
            border_style="cyan",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("Metric", style="bold white", min_width=28)
        table.add_column("Value", style="green", justify="right", min_width=16)

        # -- General --
        if "domain" in results:
            table.add_row("Domain", str(results["domain"]))
        if "segments" in results:
            table.add_row("Segments", str(results["segments"]))

        # -- Entity pipeline --
        extracted = results.get("entities_extracted", "-")
        resolved = results.get("entities_resolved", "-")
        merged = results.get("entities_merged", "-")
        kg_ent = results.get("kg_entities", "-")

        entity_flow = Text()
        entity_flow.append(str(extracted), style="white")
        entity_flow.append(" extracted", style="dim")
        if resolved != "-":
            entity_flow.append(" -> ", style="dim")
            entity_flow.append(str(resolved), style="white")
            entity_flow.append(" resolved", style="dim")
        entity_flow.append(" -> ", style="dim")
        entity_flow.append(str(kg_ent), style="bold green")
        entity_flow.append(" in KG", style="dim")
        table.add_row("Entities", entity_flow)

        if merged != "-" and merged:
            table.add_row(
                "  Duplicates Merged",
                Text(str(merged), style="yellow"),
            )

        # -- Fast first pass --
        fast_ent = results.get("fast_entities")
        fast_tri = results.get("fast_triples")
        if fast_ent or fast_tri:
            table.add_row(
                "Fast Pass (ent / tri)",
                f"{fast_ent or 0} / {fast_tri or 0}",
            )

        # -- Triple pipeline --
        tri_extracted = results.get("triples_extracted", "-")
        tri_approved = results.get("approved_triples", "-")
        tri_rejected = results.get("rejected_triples", "-")
        kg_tri = results.get("kg_triples", "-")

        triple_flow = Text()
        triple_flow.append(str(tri_extracted), style="white")
        triple_flow.append(" extracted", style="dim")
        if tri_approved != "-":
            triple_flow.append(" -> ", style="dim")
            triple_flow.append(str(tri_approved), style="white")
            triple_flow.append(" approved", style="dim")
        triple_flow.append(" -> ", style="dim")
        triple_flow.append(str(kg_tri), style="bold green")
        triple_flow.append(" in KG", style="dim")
        table.add_row("Triples", triple_flow)

        if tri_rejected != "-" and tri_rejected:
            table.add_row(
                "  Triples Rejected",
                Text(str(tri_rejected), style="yellow"),
            )

        # -- Verification --
        critic_iter = results.get("critic_iterations")
        if critic_iter is not None:
            table.add_row("Critic-Corrector Iterations", str(critic_iter))

        # -- Relation library --
        rel_lib = results.get("relation_library_size")
        if rel_lib is not None:
            table.add_row("Relation Library Size", str(rel_lib))

        # -- Timing --
        elapsed = results.get("processing_time_seconds")
        if elapsed is not None:
            table.add_row(
                "Processing Time",
                Text(_format_elapsed(elapsed), style="dim"),
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _print_summary_plain(self, results: Dict[str, Any]) -> None:
        """Render the summary as plain text."""
        print()
        print("=" * 70)
        print("PIPELINE RESULTS")
        print("=" * 70)

        if "domain" in results:
            print(f"  Domain:                       {results['domain']}")
        if "segments" in results:
            print(f"  Segments:                     {results['segments']}")

        extracted = results.get("entities_extracted", "-")
        resolved = results.get("entities_resolved", "-")
        merged = results.get("entities_merged", "-")
        kg_ent = results.get("kg_entities", "-")
        print(f"  Entities:                     {extracted} extracted", end="")
        if resolved != "-":
            print(f" -> {resolved} resolved", end="")
        print(f" -> {kg_ent} in KG")
        if merged != "-" and merged:
            print(f"    Duplicates merged:          {merged}")

        fast_ent = results.get("fast_entities")
        fast_tri = results.get("fast_triples")
        if fast_ent or fast_tri:
            print(f"  Fast pass (ent / tri):        {fast_ent or 0} / {fast_tri or 0}")

        tri_extracted = results.get("triples_extracted", "-")
        tri_approved = results.get("approved_triples", "-")
        tri_rejected = results.get("rejected_triples", "-")
        kg_tri = results.get("kg_triples", "-")
        print(f"  Triples:                      {tri_extracted} extracted", end="")
        if tri_approved != "-":
            print(f" -> {tri_approved} approved", end="")
        print(f" -> {kg_tri} in KG")
        if tri_rejected != "-" and tri_rejected:
            print(f"    Triples rejected:           {tri_rejected}")

        critic_iter = results.get("critic_iterations")
        if critic_iter is not None:
            print(f"  Critic-Corrector iterations:  {critic_iter}")

        rel_lib = results.get("relation_library_size")
        if rel_lib is not None:
            print(f"  Relation library size:        {rel_lib}")

        elapsed = results.get("processing_time_seconds")
        if elapsed is not None:
            print(f"  Processing time:              {_format_elapsed(elapsed)}")

        print("=" * 70)
        print()

    # ------------------------------------------------------------------
    # Corpus-level summary
    # ------------------------------------------------------------------

    def print_corpus_summary(self, aggregate: Dict[str, Any]) -> None:
        """Print corpus-level summary.

        Expected *aggregate* keys:
            documents_processed, total_entities, total_triples,
            total_time, memory_stats, kg_stats
        """
        if RICH_AVAILABLE:
            self._print_corpus_summary_rich(aggregate)
        else:
            self._print_corpus_summary_plain(aggregate)

    def _print_corpus_summary_rich(self, aggregate: Dict[str, Any]) -> None:
        """Render corpus summary as a Rich table."""
        table = Table(
            title="Corpus Processing Complete",
            box=box.DOUBLE,
            title_style="bold white",
            border_style="green",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("Metric", style="bold white", min_width=28)
        table.add_column("Value", style="green", justify="right", min_width=16)

        docs = aggregate.get("documents_processed", 0)
        table.add_row("Documents Processed", str(docs))
        table.add_row("Total Entities", str(aggregate.get("total_entities", 0)))
        table.add_row("Total Triples", str(aggregate.get("total_triples", 0)))

        total_time = aggregate.get("total_time", 0)
        table.add_row(
            "Total Processing Time",
            Text(_format_elapsed(total_time), style="dim"),
        )

        if docs > 0 and total_time > 0:
            avg = total_time / docs
            table.add_row(
                "Avg Time per Document",
                Text(_format_elapsed(avg), style="dim"),
            )

        # KG stats
        kg_stats = aggregate.get("kg_stats", {})
        if kg_stats:
            table.add_row(
                "KG Total Entities",
                str(kg_stats.get("total_entities", "-")),
            )
            table.add_row(
                "KG Total Triples",
                str(kg_stats.get("total_triples", "-")),
            )

        # Memory stats
        mem_stats = aggregate.get("memory_stats", {})
        if mem_stats:
            aliases = mem_stats.get("entity_aliases", 0)
            unique = mem_stats.get("unique_entities", 0)
            if aliases or unique:
                table.add_row("Entity Aliases", str(aliases))
                table.add_row("Unique Entities Tracked", str(unique))

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _print_corpus_summary_plain(self, aggregate: Dict[str, Any]) -> None:
        """Render corpus summary as plain text."""
        print()
        print("=" * 70)
        print("CORPUS PROCESSING COMPLETE")
        print("=" * 70)

        docs = aggregate.get("documents_processed", 0)
        print(f"  Documents processed:          {docs}")
        print(f"  Total entities:               {aggregate.get('total_entities', 0)}")
        print(f"  Total triples:                {aggregate.get('total_triples', 0)}")

        total_time = aggregate.get("total_time", 0)
        print(f"  Total processing time:        {_format_elapsed(total_time)}")

        if docs > 0 and total_time > 0:
            avg = total_time / docs
            print(f"  Avg time per document:        {_format_elapsed(avg)}")

        kg_stats = aggregate.get("kg_stats", {})
        if kg_stats:
            print(f"  KG total entities:            {kg_stats.get('total_entities', '-')}")
            print(f"  KG total triples:             {kg_stats.get('total_triples', '-')}")

        mem_stats = aggregate.get("memory_stats", {})
        if mem_stats:
            aliases = mem_stats.get("entity_aliases", 0)
            unique = mem_stats.get("unique_entities", 0)
            if aliases or unique:
                print(f"  Entity aliases:               {aliases}")
                print(f"  Unique entities tracked:      {unique}")

        print("=" * 70)
        print()

    # ------------------------------------------------------------------
    # Utility: info / warning / error messages
    # ------------------------------------------------------------------

    def info(self, message: str) -> None:
        """Print an informational message."""
        if RICH_AVAILABLE:
            self.console.print(f"  [dim]i[/dim] {message}")
        else:
            print(f"  i {message}")

    def warning(self, message: str) -> None:
        """Print a warning message."""
        if RICH_AVAILABLE:
            self.console.print(f"  [yellow]![/yellow] [yellow]{message}[/yellow]")
        else:
            print(f"  ! WARNING: {message}")

    def error(self, message: str) -> None:
        """Print an error message."""
        if RICH_AVAILABLE:
            self.console.print(f"  [red bold]X[/red bold] [red]{message}[/red]")
        else:
            print(f"  X ERROR: {message}")
