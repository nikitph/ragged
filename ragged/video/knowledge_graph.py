import json
import struct
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import spacy
import networkx as nx
from collections import defaultdict, Counter
import hashlib
from datetime import datetime


@dataclass
class Entity:
    """Represents an extracted entity with metadata"""
    text: str
    label: str
    start_pos: int
    end_pos: int
    chunk_id: int
    confidence: float = 1.0
    entity_id: Optional[str] = None

    def __post_init__(self):
        if self.entity_id is None:
            self.entity_id = hashlib.md5(f"{self.text}_{self.label}".encode()).hexdigest()[:8]


@dataclass
class Relation:
    """Represents a relationship between entities"""
    source_entity: str
    target_entity: str
    relation_type: str
    context: str
    chunk_id: int
    confidence: float = 1.0
    relation_id: Optional[str] = None

    def __post_init__(self):
        if self.relation_id is None:
            rel_str = f"{self.source_entity}_{self.relation_type}_{self.target_entity}"
            self.relation_id = hashlib.md5(rel_str.encode()).hexdigest()[:8]


@dataclass
class KnowledgeGraphFragment:
    """Represents a knowledge graph fragment with entities and relations"""
    fragment_id: int
    entities: List[Entity]
    relations: List[Relation]
    chunk_ids: Set[int]
    entity_count: int = 0
    relation_count: int = 0
    timestamp: str = ""

    def __post_init__(self):
        self.entity_count = len(self.entities)
        self.relation_count = len(self.relations)
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# Interfaces for SOLID principles
class EntityExtractor(Protocol):
    """Interface for entity extraction strategies"""

    def extract_entities(self, text: str, chunk_id: int) -> List[Entity]:
        """Extract entities from text"""
        ...


class RelationExtractor(Protocol):
    """Interface for relation extraction strategies"""

    def extract_relations(self, text: str, entities: List[Entity], chunk_id: int) -> List[Relation]:
        """Extract relations between entities from text"""
        ...


class GraphBuilder(Protocol):
    """Interface for building knowledge graphs"""

    def build_graph(self, entities: List[Entity], relations: List[Relation]) -> nx.Graph:
        """Build NetworkX graph from entities and relations"""
        ...


class GraphFragmenter(Protocol):
    """Interface for fragmenting knowledge graphs"""

    def create_fragments(self, graph: nx.Graph, entities: List[Entity],
                         relations: List[Relation], max_fragment_size: int) -> List[KnowledgeGraphFragment]:
        """Create knowledge graph fragments"""
        ...


# Concrete implementations
class SpacyEntityExtractor:
    """Extract entities using spaCy NLP"""

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")

    def extract_entities(self, text: str, chunk_id: int) -> List[Entity]:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP"}:
                entity = Entity(
                    text=ent.text.strip(),
                    label=ent.label_,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    chunk_id=chunk_id,
                    confidence=1.0  # spaCy doesn't provide confidence scores by default
                )
                entities.append(entity)

        return entities


class PatternRelationExtractor:
    """Extract relations using pattern matching"""

    def __init__(self):
        self.relation_patterns = {
            "works_at": ["works at", "employed by", "employee of"],
            "founded": ["founded", "established", "created"],
            "located_in": ["located in", "based in", "headquartered in"],
            "part_of": ["part of", "division of", "subsidiary of"],
            "develops": ["develops", "creates", "builds", "makes"],
            "uses": ["uses", "utilizes", "employs", "applies"]
        }

    def extract_relations(self, text: str, entities: List[Entity], chunk_id: int) -> List[Relation]:
        """Extract relations using pattern matching between entities"""
        relations = []
        text_lower = text.lower()

        # Sort entities by position for context window analysis
        sorted_entities = sorted(entities, key=lambda e: e.start_pos)

        for i, source_entity in enumerate(sorted_entities):
            for j, target_entity in enumerate(sorted_entities[i + 1:], i + 1):
                # Look for relations in the context between entities
                context_start = source_entity.end_pos
                context_end = target_entity.start_pos

                if context_end - context_start > 200:  # Skip if entities are too far apart
                    continue

                context = text[context_start:context_end].lower()

                for relation_type, patterns in self.relation_patterns.items():
                    for pattern in patterns:
                        if pattern in context:
                            relation = Relation(
                                source_entity=source_entity.entity_id,
                                target_entity=target_entity.entity_id,
                                relation_type=relation_type,
                                context=text[source_entity.start_pos:target_entity.end_pos],
                                chunk_id=chunk_id,
                                confidence=0.8  # Pattern-based confidence
                            )
                            relations.append(relation)
                            break

        return relations


class NetworkXGraphBuilder:
    """Build NetworkX graphs from entities and relations"""

    def build_graph(self, entities: List[Entity], relations: List[Relation]) -> nx.Graph:
        """Build a NetworkX graph from entities and relations"""
        graph = nx.Graph()

        # Add entity nodes
        for entity in entities:
            graph.add_node(
                entity.entity_id,
                text=entity.text,
                label=entity.label,
                chunk_ids={entity.chunk_id},
                confidence=entity.confidence
            )

        # Add relation edges
        for relation in relations:
            if graph.has_node(relation.source_entity) and graph.has_node(relation.target_entity):
                # Merge if edge exists, otherwise create new
                if graph.has_edge(relation.source_entity, relation.target_entity):
                    edge_data = graph[relation.source_entity][relation.target_entity]
                    if "relations" in edge_data and hasattr(edge_data["relations"], 'append'):
                        edge_data["relations"].append({
                            "type": relation.relation_type,
                            "context": relation.context,
                            "chunk_id": relation.chunk_id,
                            "confidence": relation.confidence
                        })
                    else:
                        # Initialize relations list if it doesn't exist or is wrong type
                        edge_data["relations"] = [{
                            "type": relation.relation_type,
                            "context": relation.context,
                            "chunk_id": relation.chunk_id,
                            "confidence": relation.confidence
                        }]
                else:
                    graph.add_edge(
                        relation.source_entity,
                        relation.target_entity,
                        relations=[{
                            "type": relation.relation_type,
                            "context": relation.context,
                            "chunk_id": relation.chunk_id,
                            "confidence": relation.confidence
                        }]
                    )

        return graph


class CommunityGraphFragmenter:
    """Fragment graphs using community detection"""

    def create_fragments(self, graph: nx.Graph, entities: List[Entity],
                         relations: List[Relation], max_fragment_size: int) -> List[KnowledgeGraphFragment]:
        """Create fragments using community detection and size constraints"""
        if len(graph.nodes) == 0:
            return []

        # Use simple connected components for fragmentation
        fragments = []
        fragment_id = 0

        # Get connected components
        components = list(nx.connected_components(graph))

        for component in components:
            component_nodes = list(component)

            # If component is too large, split it
            if len(component_nodes) > max_fragment_size:
                # Split large components into smaller chunks
                for i in range(0, len(component_nodes), max_fragment_size):
                    chunk_nodes = component_nodes[i:i + max_fragment_size]
                    fragment = self._create_fragment_from_nodes(
                        graph, chunk_nodes, entities, relations, fragment_id
                    )
                    fragments.append(fragment)
                    fragment_id += 1
            else:
                # Use component as-is
                fragment = self._create_fragment_from_nodes(
                    graph, component_nodes, entities, relations, fragment_id
                )
                fragments.append(fragment)
                fragment_id += 1

        return fragments

    def _create_fragment_from_nodes(self, graph: nx.Graph, nodes: List[str],
                                    entities: List[Entity], relations: List[Relation],
                                    fragment_id: int) -> KnowledgeGraphFragment:
        """Create a fragment from a set of nodes"""
        node_set = set(nodes)

        # Filter entities that belong to this fragment
        fragment_entities = [e for e in entities if e.entity_id in node_set]

        # Filter relations that belong to this fragment
        fragment_relations = [
            r for r in relations
            if r.source_entity in node_set and r.target_entity in node_set
        ]

        # Get chunk IDs from entities and relations
        chunk_ids = set()
        for entity in fragment_entities:
            chunk_ids.add(entity.chunk_id)
        for relation in fragment_relations:
            chunk_ids.add(relation.chunk_id)

        return KnowledgeGraphFragment(
            fragment_id=fragment_id,
            entities=fragment_entities,
            relations=fragment_relations,
            chunk_ids=chunk_ids
        )


class KnowledgeGraphTrack:
    """Main class for knowledge graph track functionality"""

    def __init__(self,
                 entity_extractor: EntityExtractor = None,
                 relation_extractor: RelationExtractor = None,
                 graph_builder: GraphBuilder = None,
                 graph_fragmenter: GraphFragmenter = None,
                 max_fragment_size: int = 100):
        # Dependency injection with defaults
        self.entity_extractor = entity_extractor or SpacyEntityExtractor()
        self.relation_extractor = relation_extractor or PatternRelationExtractor()
        self.graph_builder = graph_builder or NetworkXGraphBuilder()
        self.graph_fragmenter = graph_fragmenter or CommunityGraphFragmenter()
        self.max_fragment_size = max_fragment_size

        self.all_entities: List[Entity] = []
        self.all_relations: List[Relation] = []
        self.graph: Optional[nx.Graph] = None
        self.fragments: List[KnowledgeGraphFragment] = []

    def process_text_chunks(self, text_chunks: List['TextChunk']) -> List[KnowledgeGraphFragment]:
        """Process text chunks to extract knowledge graph fragments"""
        print(f"Processing {len(text_chunks)} text chunks for knowledge graph extraction...")

        # Extract entities and relations from each chunk
        for chunk in text_chunks:
            entities = self.entity_extractor.extract_entities(chunk.text, chunk.chunk_id)
            relations = self.relation_extractor.extract_relations(chunk.text, entities, chunk.chunk_id)

            self.all_entities.extend(entities)
            self.all_relations.extend(relations)

        print(f"Extracted {len(self.all_entities)} entities and {len(self.all_relations)} relations")

        # Build global graph
        self.graph = self.graph_builder.build_graph(self.all_entities, self.all_relations)
        print(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")

        # Create fragments
        self.fragments = self.graph_fragmenter.create_fragments(
            self.graph, self.all_entities, self.all_relations, self.max_fragment_size
        )

        print(f"Created {len(self.fragments)} knowledge graph fragments")
        return self.fragments

    def serialize_fragment(self, fragment: KnowledgeGraphFragment) -> bytes:
        """Serialize a knowledge graph fragment to bytes"""
        fragment_data = {
            "fragment_id": fragment.fragment_id,
            "entity_count": fragment.entity_count,
            "relation_count": fragment.relation_count,
            "chunk_ids": list(fragment.chunk_ids),
            "timestamp": fragment.timestamp,
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "text": e.text,
                    "label": e.label,
                    "chunk_id": e.chunk_id,
                    "confidence": e.confidence
                }
                for e in fragment.entities
            ],
            "relations": [
                {
                    "relation_id": r.relation_id,
                    "source_entity": r.source_entity,
                    "target_entity": r.target_entity,
                    "relation_type": r.relation_type,
                    "context": r.context[:200],  # Truncate context for size
                    "chunk_id": r.chunk_id,
                    "confidence": r.confidence
                }
                for r in fragment.relations
            ]
        }

        # Serialize to JSON bytes
        json_data = json.dumps(fragment_data, separators=(',', ':')).encode('utf-8')

        # Create header with size information
        header = struct.pack('<III',
                             len(json_data),  # JSON data size
                             fragment.entity_count,  # Entity count
                             fragment.relation_count)  # Relation count

        return header + json_data

    def get_track_metadata(self) -> Dict[str, Any]:
        """Get metadata for the knowledge graph track"""
        return {
            "track_type": "knowledge_graph",
            "total_entities": len(self.all_entities),
            "total_relations": len(self.all_relations),
            "total_fragments": len(self.fragments),
            "graph_stats": {
                "nodes": len(self.graph.nodes) if self.graph else 0,
                "edges": len(self.graph.edges) if self.graph else 0,
                "connected_components": nx.number_connected_components(self.graph) if self.graph else 0
            },
            "entity_types": dict(Counter(e.label for e in self.all_entities)),
            "relation_types": dict(Counter(r.relation_type for r in self.all_relations)),
            "fragments": [
                {
                    "fragment_id": f.fragment_id,
                    "entity_count": f.entity_count,
                    "relation_count": f.relation_count,
                    "chunk_ids": list(f.chunk_ids)
                }
                for f in self.fragments
            ]
        }


# Integration interface for the main MP4 encoder
class KnowledgeGraphIntegration:
    """Integration layer for adding knowledge graph track to MP4 encoder"""

    def __init__(self, knowledge_graph_track: KnowledgeGraphTrack):
        self.kg_track = knowledge_graph_track

    def process_and_integrate(self, text_chunks: List['TextChunk']) -> Tuple[List[bytes], Dict[str, Any]]:
        """Process text chunks and return serialized KG fragments with metadata"""
        # Process chunks to create KG fragments
        kg_fragments = self.kg_track.process_text_chunks(text_chunks)

        # Serialize fragments
        serialized_fragments = []
        for fragment in kg_fragments:
            serialized_data = self.kg_track.serialize_fragment(fragment)
            serialized_fragments.append(serialized_data)

        # Get track metadata
        track_metadata = self.kg_track.get_track_metadata()

        return serialized_fragments, track_metadata