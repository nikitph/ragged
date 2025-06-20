"""
Real-world test using electrophoresis document
Tests text-to-MP4 pipeline with actual scientific content
"""

import pytest
import tempfile
import json
from pathlib import Path
import numpy as np

from ragged.video.encoder import TextVectorPipeline, VectorMP4Encoder, create_text_vector_mp4


class TestElectrophoresisDocument:
    """Test with real scientific document content"""

    @pytest.fixture
    def electrophoresis_text(self):
        """Real electrophoresis document content"""
        return """Electrophoresis is a fundamental technique used in biochemistry, molecular biology, and analytical chemistry to separate charged biomolecules—like DNA, RNA, or proteins—based on their size, charge, and conformation under the influence of an electric field.

🔬 Definition

Electrophoresis is the movement of charged particles through a medium (usually a gel or liquid) in response to an electric field. Positively charged particles migrate toward the cathode (−), while negatively charged ones move toward the anode (+).

📚 Types of Electrophoresis

Agarose Gel Electrophoresis is used for DNA/RNA separation using agarose gel medium.
Polyacrylamide Gel Electrophoresis (PAGE) separates proteins and DNA fragments using polyacrylamide gel.
SDS-PAGE is used for denatured proteins with SDS-treated polyacrylamide.
Isoelectric Focusing (IEF) separates proteins by pI using pH gradient gel.
Capillary Electrophoresis handles small ions and peptides in capillary tubes.
2D Electrophoresis processes complex protein mixtures using IEF plus SDS-PAGE.

⚙️ Principle

Charge-Based Separation: Particles move in response to an electric field. The direction and speed depend on net charge, size, shape, and medium resistance.

Gel Matrix: Acts like a sieve. Larger molecules move slower through the pores.

Electric Field: Typically 50–150 volts is applied. Higher voltages speed up migration but can cause smearing due to heat.

🧪 Common Buffers

TAE or TBE buffers are used in agarose for DNA to maintain pH and conduct current.
Tris-Glycine buffer is used in SDS-PAGE to provide ions and stable pH.
Urea buffer is used in IEF or PAGE as a denaturing agent for protein unfolding.

🧬 Application Areas

Nucleic Acid Analysis includes DNA fingerprinting for forensics, PCR verification, plasmid mapping, and RNA integrity checking.

Protein Analysis covers protein purity assessment, molecular weight determination, Western blotting with SDS-PAGE plus antibody probing, and 2D electrophoresis for proteomics.

Clinical Diagnostics involves hemoglobin variants like Sickle Cell detection and serum protein electrophoresis for multiple myeloma.

🧬 Step-by-Step: Agarose Gel Electrophoresis

Gel Preparation: Agarose is melted in buffer, poured into a tray, and allowed to set with a comb for wells.
Sample Preparation: DNA mixed with loading dye adds weight and color.
Running the Gel: Samples are loaded into wells, and current is applied at around 100V.
Migration: DNA moves toward the anode due to negative charge. Shorter fragments move faster.
Visualization: Gel is stained with ethidium bromide or SYBR Green, then visualized under UV light.

🔍 SDS-PAGE for Protein Separation

SDS Denaturation: Proteins are treated with Sodium Dodecyl Sulfate to impart uniform negative charge and unfold 3D structure.
Gel Loading: Protein samples with dye are loaded into wells.
Electrophoresis: Proteins migrate based on molecular weight with small proteins moving faster.
Staining: Gel is stained with Coomassie Blue or Silver stain to visualize protein bands.

⚠️ Factors Affecting Electrophoresis

Voltage affects speed - higher voltage means faster migration but may cause overheating.
Gel concentration - higher percentage gels provide better separation of small molecules.
Buffer composition affects migration speed and resolution.
Sample quality - degraded samples smear or show extra bands.
Time of run - under or over-running may distort bands.

✅ Advantages

Electrophoresis is simple and cost-effective with high resolution of biomolecules.
Can separate by multiple properties including charge, size, and pI.
Adaptable for many different molecules.

❌ Limitations

Not quantitative unless combined with densitometry.
Limited scalability and sensitive to run conditions.
Smearing and artifacts can occur with improper setup."""

    def test_real_document_processing(self, electrophoresis_text, tmp_path):
        """Test processing real scientific document"""
        pipeline = TextVectorPipeline(
            model_name="all-MiniLM-L6-v2",
            chunk_size=200,
            chunk_overlap=30,
            vector_dim=384
        )

        doc = {"text": electrophoresis_text, "source": "electrophoresis_guide.txt"}
        vectors, metadata = pipeline.process_documents([doc])

        # Should create meaningful chunks
        assert len(vectors) > 5  # Expect multiple chunks from this content
        assert vectors.shape[1] == 384

        # Check topic extraction worked
        topics = [meta.get("topic") for meta in metadata]
        meaningful_topics = [t for t in topics if t and len(t) > 2]
        assert len(meaningful_topics) > 0

        print(f"Generated {len(vectors)} chunks with topics: {set(meaningful_topics)}")

    def test_scientific_vocabulary_chunking(self, electrophoresis_text, tmp_path):
        """Test chunking preserves scientific terminology"""
        pipeline = TextVectorPipeline(chunk_size=150, chunk_overlap=25)
        chunks = pipeline.chunk_text(electrophoresis_text, "science.txt")

        # Should preserve key scientific terms across chunks
        key_terms = ["electrophoresis", "DNA", "protein", "gel", "SDS-PAGE"]

        for term in key_terms:
            found_in_chunks = sum(1 for chunk in chunks if term.lower() in chunk.text.lower())
            assert found_in_chunks > 0, f"Term '{term}' not found in any chunk"

    def test_complete_pipeline_with_real_data(self, electrophoresis_text, tmp_path):
        """Test end-to-end pipeline with scientific content"""
        documents = [
            {"text": electrophoresis_text, "source": "electrophoresis.txt"},
            {
                "text": "PCR amplification is a molecular biology technique used to amplify DNA sequences. The process uses thermal cycling with DNA polymerase enzyme.",
                "source": "pcr_basics.txt"
            },
            {
                "text": "Western blotting combines gel electrophoresis with immunodetection to identify specific proteins in samples using antibodies.",
                "source": "western_blot.txt"
            }
        ]

        output_file = tmp_path / "scientific_knowledge.mp4"

        # Should process without errors
        create_text_vector_mp4(documents, str(output_file))

        # Verify outputs
        assert output_file.exists()
        assert output_file.stat().st_size > 1000  # Should be substantial

        # Check manifest
        manifest_file = tmp_path / "scientific_knowledge_manifest.json"
        assert manifest_file.exists()

        with open(manifest_file, 'r') as f:
            manifest = json.load(f)

        assert manifest["metadata"]["total_vectors"] >= 4
        print(f"Created {manifest['metadata']['total_vectors']} vectors from scientific documents")

    def test_search_quality_with_real_content(self, tmp_path):
        """Test search quality using real scientific content"""
        # Create knowledge base with related but distinct topics
        knowledge_base = [
            {
                "text": "Agarose gel electrophoresis separates DNA fragments by size. Smaller fragments migrate faster through gel pores.",
                "source": "dna_separation.txt"
            },
            {
                "text": "SDS-PAGE denatures proteins and separates them by molecular weight using polyacrylamide gel matrix.",
                "source": "protein_separation.txt"
            },
            {
                "text": "PCR uses thermal cycling to amplify specific DNA sequences using DNA polymerase and primers.",
                "source": "pcr_method.txt"
            },
            {
                "text": "Western blotting detects specific proteins using antibodies after electrophoretic separation.",
                "source": "protein_detection.txt"
            }
        ]

        # Process through pipeline
        pipeline = TextVectorPipeline()
        vectors, metadata = pipeline.process_documents(knowledge_base)

        # Encode to MP4
        encoder = VectorMP4Encoder(vector_dim=384, chunk_size=10)
        encoder.add_vectors(vectors, metadata)

        output_file = tmp_path / "search_test.mp4"
        encoder.encode_to_mp4(str(output_file))

        # Test search functionality would go here
        # (requires decoder implementation)

        # For now, verify meaningful vector similarities
        if len(vectors) >= 2:
            # DNA-related chunks should be more similar to each other
            similarities = np.dot(vectors, vectors.T)

            # Check that we have varying similarities (not all identical)
            unique_sims = len(set(similarities[0, 1:].round(3)))
            assert unique_sims > 1, "All vectors too similar - may indicate poor chunking"

            print(f"Vector similarities range: {similarities.min():.3f} to {similarities.max():.3f}")

    def test_topic_extraction_quality(self, electrophoresis_text):
        """Test topic extraction produces meaningful results"""
        pipeline = TextVectorPipeline(chunk_size=100)
        chunks = pipeline.chunk_text(electrophoresis_text, "topics_test.txt")

        # Extract topics from chunks
        topics = [chunk.topic for chunk in chunks if chunk.topic]

        # Should find relevant scientific topics
        expected_topics = ["electrophoresis", "protein", "separation", "gel", "analysis"]
        found_relevant = any(
            any(expected in topic.lower() for expected in expected_topics)
            for topic in topics if topic
        )

        assert found_relevant, f"No relevant topics found. Got: {topics}"
        print(f"Extracted topics: {set(topics)}")

    def test_metadata_preservation(self, electrophoresis_text):
        """Test metadata is properly preserved through pipeline"""
        pipeline = TextVectorPipeline()
        doc = {"text": electrophoresis_text, "source": "metadata_test.txt"}
        vectors, metadata = pipeline.process_documents([doc])

        # Check all required metadata fields
        required_fields = ["text", "source", "chunk_id", "word_count", "token_count", "text_hash"]

        for meta in metadata:
            for field in required_fields:
                assert field in meta, f"Missing field: {field}"

            # Verify data quality
            assert meta["word_count"] > 0
            assert meta["token_count"] > 0
            assert len(meta["text_hash"]) == 8  # MD5 hash truncated
            assert meta["source"] == "metadata_test.txt"

    def test_unicode_and_special_characters(self):
        """Test handling of scientific unicode and special characters"""
        special_text = """
        Electrophoresis uses electric field (→) to separate molecules.
        Temperature range: 4°C to 25°C is optimal.
        Buffer concentration: 1× TAE (Tris-acetate-EDTA).
        Voltage: 50–150V typically applied.
        Molecular weights: 100 kDa ± 5% accuracy.
        pH range: 7.0 ≤ pH ≤ 8.5 for stability.
        """

        pipeline = TextVectorPipeline()
        doc = {"text": special_text, "source": "unicode_test.txt"}

        # Should handle without errors
        vectors, metadata = pipeline.process_documents([doc])
        assert len(vectors) > 0

        # Check that special characters are preserved in metadata
        combined_text = " ".join(meta["text"] for meta in metadata)
        assert "°C" in combined_text or "TAE" in combined_text


class TestVectorMP4Decoder:
    """Test MP4 decoding and search functionality"""

    def test_complete_roundtrip(self, tmp_path):
        """Test encode → decode → search roundtrip"""
        # Create test documents
        docs = [
            {"text": "DNA electrophoresis separates nucleic acids by size using agarose gel matrix.", "source": "dna.txt"},
            {"text": "Protein separation uses SDS-PAGE with polyacrylamide gel for molecular weight analysis.", "source": "protein.txt"},
            {"text": "PCR amplification requires DNA polymerase, primers, and thermal cycling conditions.", "source": "pcr.txt"}
        ]

        # Encode to MP4
        output_file = tmp_path / "roundtrip_test.mp4"
        create_text_vector_mp4(docs, str(output_file))

        # Import decoder
        from decoder import VectorMP4Decoder

        # Decode and test
        decoder = VectorMP4Decoder(str(output_file))

        # Test vector retrieval by ID
        vectors, metadata = decoder.get_vectors_by_ids([0, 1])
        assert len(vectors) >= 1
        assert len(metadata) >= 1

        # Test manifest access
        assert decoder.manifest["metadata"]["total_vectors"] >= 3
        print(f"Successfully decoded {decoder.manifest['metadata']['total_vectors']} vectors")

    def test_search_functionality(self, tmp_path):
        """Test vector search with real queries"""
        # Create knowledge base
        docs = [
            {"text": "Agarose gel electrophoresis is used for DNA fragment separation and analysis.", "source": "dna_method.txt"},
            {"text": "SDS-PAGE protein electrophoresis denatures proteins for molecular weight determination.", "source": "protein_method.txt"},
            {"text": "Western blotting combines electrophoresis with antibody detection for protein identification.", "source": "western.txt"}
        ]

        # Encode
        output_file = tmp_path / "search_test.mp4"
        create_text_vector_mp4(docs, str(output_file))

        # Decode and search
        from encoder import TextVectorPipeline
        from decoder import VectorMP4Decoder

        decoder = VectorMP4Decoder(str(output_file))
        pipeline = TextVectorPipeline()

        # Test search with DNA-related query
        query_text = "DNA separation and fragment analysis"
        query_vector = pipeline.model.encode([query_text])[0]

        results = decoder.search_vectors(query_vector, top_k=2)

        assert len(results) > 0
        assert all("similarity" in result for result in results)

        # Should find DNA-related content
        dna_found = any("DNA" in result["metadata"].get("text", "") for result in results)
        assert dna_found

        print(f"Search found {len(results)} results for DNA query")
        for i, result in enumerate(results):
            text_preview = result["metadata"]["text"][:50] + "..."
            print(f"  {i+1}. {text_preview} (sim: {result['similarity']:.3f})")

    def test_topic_filtering(self, tmp_path):
        """Test topic-based vector retrieval"""
        docs = [
            {"text": "Electrophoresis principles involve charged particle migration in electric fields.", "source": "principles.txt"},
            {"text": "Laboratory protocols require proper buffer preparation and gel casting techniques.", "source": "protocols.txt"}
        ]

        output_file = tmp_path / "topic_test.mp4"
        create_text_vector_mp4(docs, str(output_file))

        from decoder import VectorMP4Decoder
        decoder = VectorMP4Decoder(str(output_file))

        # Get all available topics
        all_vectors = []
        all_metadata = []
        for frag_info in decoder.manifest["metadata"]["fragments"]:
            fragment = decoder._read_fragment(frag_info["id"])
            all_vectors.extend(fragment["vectors"])
            all_metadata.extend(fragment["metadata"])

        topics = {meta.get("topic") for meta in all_metadata if meta.get("topic")}
        print(f"Available topics: {topics}")

        # Test topic retrieval if topics exist
        if topics:
            test_topic = list(topics)[0]
            topic_vectors, topic_metadata = decoder.get_vectors_by_topic(test_topic)

            if len(topic_vectors) > 0:
                assert all(meta.get("topic") == test_topic for meta in topic_metadata)
                print(f"Retrieved {len(topic_vectors)} vectors for topic '{test_topic}'")

    def test_vector_integrity(self, tmp_path):
        """Test that decoded vectors match original embeddings"""
        docs = [{"text": "Test document for vector integrity validation.", "source": "integrity.txt"}]

        # Generate original vectors
        pipeline = TextVectorPipeline()
        original_vectors, original_metadata = pipeline.process_documents(docs)

        # Encode to MP4
        output_file = tmp_path / "integrity_test.mp4"
        encoder = VectorMP4Encoder(vector_dim=384)
        encoder.add_vectors(original_vectors, original_metadata)
        encoder.encode_to_mp4(str(output_file))

        # Decode and compare
        from decoder import VectorMP4Decoder
        decoder = VectorMP4Decoder(str(output_file))

        decoded_vectors, decoded_metadata = decoder.get_vectors_by_ids(list(range(len(original_vectors))))

        # Check vector similarity (should be nearly identical)
        if len(decoded_vectors) > 0:
            similarity = np.dot(original_vectors[0], decoded_vectors[0])
            assert similarity > 0.99, f"Vector similarity too low: {similarity}"
            print(f"Vector integrity check passed: similarity = {similarity:.6f}")


if __name__ == "__main__":
    # Run real content tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])