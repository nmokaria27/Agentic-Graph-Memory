# Multi-Agent Knowledge Graph Enrichment Framework

A framework for building knowledge graphs using multiple LLM-backed agents. The system uses a collaborative multi-agent architecture where specialized agents handle different aspects of knowledge extraction: ingestion, segmentation, entity extraction, relation extraction, conflict detection, and verification. 

A key feature is the shared memory and blackboard system that enables cross-document reasoning. The SharedMemory module maintains episodic memory (document-specific context), semantic memory (persistent facts), and working memory (current processing state). The blackboard serves as a shared scratchpad where agents can post hypotheses, partial results, and requests for collaboration. This allows agents to build upon each other's findings and maintain context across multiple documents being processed together.

Agents communicate through a MessageBus using a defined collaboration protocol. Agents can send direct messages, broadcast to all agents, or request specific capabilities from others. The protocol supports message types like proposals, validations, and conflict reports, enabling agents to coordinate on ambiguous extractions or request help when confidence is low.

The framework is domain-agnostic and supports open-world relation extraction, meaning it can discover new relation types not defined in advance. Still early days, more work to come.