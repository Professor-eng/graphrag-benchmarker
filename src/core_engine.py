import os
import re
import json
import time
import chromadb
import pyTigerGraph as tg
from google import genai
from dotenv import load_dotenv

load_dotenv()

class TigerGraphOfficialGraphRAG:
    def __init__(self, target_graph: str = "CyberSecurity_Network_Graph"):
        raw_host = os.getenv("TIGERGRAPH_HOST", "http://127.0.0.1")
        clean_host = raw_host.replace("https://", "").replace("http://", "").split("/")[0]
        
        # Real-time State Connection Matrix
        self.conn = tg.TigerGraphConnection(
            host=f"http://{clean_host}" if "127.0.0.1" in clean_host or "localhost" in clean_host else f"https://{clean_host}",
            graphname=target_graph,
            username=os.getenv("TG_USERNAME", "tigergraph"),
            password=os.getenv("TG_PASSWORD", "tigergraph"),
            apiToken=os.getenv("TIGERGRAPH_TOKEN")
        )
        
        # Test connection immediately to avoid silent failures
        try:
            self.conn.ping()
            print(f"Successfully verified active link to TigerGraph graph: {target_graph}")
        except Exception as e:
            print(f"WARNING: TigerGraph connection failed on startup. Errors: {str(e)}")
        
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = "gemini-2.5-flash"
        
        self.chroma_client = chromadb.Client()
        try:
            self.vector_store = self.chroma_client.get_collection(name="tg_structural_chunks")
        except Exception:
            self.vector_store = self.chroma_client.create_collection(name="tg_structural_chunks")

    def initialize_production_schema(self):
        """Forces schema vertex and edge layout generation inside the active graph container context."""
        try:
            # Drop query down the REST++ pipeline channel
            schema_gsql = """
            CREATE VERTEX DocChunk(id STRING, content STRING, tokens INT) WITH primary_id_as_attribute="true";
            CREATE VERTEX ConceptNode(id STRING, label_type STRING, structural_summary STRING) WITH primary_id_as_attribute="true";
            CREATE DIRECTED EDGE SIBLING_LINK(FROM DocChunk, TO DocChunk, sequence_weight INT);
            CREATE DIRECTED EDGE CHUNK_EXPOSES_CONCEPT(FROM DocChunk, TO ConceptNode);
            CREATE DIRECTED EDGE STRUCTURAL_RELATION(FROM ConceptNode, TO ConceptNode, verb_type STRING, dynamic_weight FLOAT);
            """
            current_schema = self.conn.getSchema()
            if "DocChunk" not in str(current_schema):
                # Only try creating if vertices are absent from current graph definitions
                self.conn.gsql(f"USE GRAPH {self.conn.graphname}\n" + schema_gsql)
        except Exception as e:
            print(f"Schema status logging verification parameter: {str(e)}")

    def extract_entities_and_upsert_graph(self, corpus_text: str, batch_index: int = 0):
        """Processes source text and inserts structural vertices/edges directly into TigerGraph."""
        self.initialize_production_schema()
        
        words = corpus_text.split()
        chunk_size = 1000
        overlap = 150
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            segment = " ".join(words[i:i + chunk_size])
            chunks.append(segment)
            if i + chunk_size >= len(words):
                break

        for idx, text_block in enumerate(chunks):
            chunk_uuid = f"chunk_{batch_index}_{idx}_{int(time.time())}"
            computed_tokens = len(text_block.split())
            
            # Sync to local spatial benchmark cache
            self.vector_store.add(
                documents=[text_block],
                ids=[chunk_uuid],
                metadatas=[{"batch": batch_index, "index": idx}]
            )
            
            # Write to active TigerGraph engine storage structures
            self.conn.upsertVertex("DocChunk", chunk_uuid, attributes={"content": text_block, "tokens": computed_tokens})
            
            if idx > 0 and 'prev_uuid' in locals():
                self.conn.upsertEdge("DocChunk", prev_uuid, "SIBLING_LINK", "DocChunk", chunk_uuid, attributes={"sequence_weight": 1})
            prev_uuid = chunk_uuid

            # Structured Extraction Generation Layer
            extraction_prompt = f"""
            Extract entities and explicit direct relationships from this text block.
            Return a JSON array of objects with keys: 'source_node', 'source_type', 'relation', 'target_node', 'target_type', 'contextual_summary'.
            Text: {text_block}
            """
            try:
                raw_response = self.client.models.generate_content(model=self.model_name, contents=extraction_prompt).text
                clean_json = raw_response.strip().replace("```json", "").replace("```", "")
                relationships = json.loads(clean_json)
                
                for link in relationships:
                    src = re.sub(r'\W+', '_', str(link['source_node']))
                    tgt = re.sub(r'\W+', '_', str(link['target_node']))
                    verb = re.sub(r'\W+', '_', str(link['relation'])).upper()
                    
                    self.conn.upsertVertex("ConceptNode", src, attributes={"label_type": link['source_type'], "structural_summary": link['contextual_summary']})
                    self.conn.upsertVertex("ConceptNode", tgt, attributes={"label_type": link['target_type'], "structural_summary": link['contextual_summary']})
                    
                    self.conn.upsertEdge("DocChunk", chunk_uuid, "CHUNK_EXPOSES_CONCEPT", "ConceptNode", src)
                    self.conn.upsertEdge("ConceptNode", src, "STRUCTURAL_RELATION", "ConceptNode", tgt, attributes={"verb_type": verb, "dynamic_weight": 1.0})
            except Exception as e:
                print(f"Skipping malformed payload item block: {str(e)}")
                continue

    def run_graph_rag_pipeline(self, operational_query: str) -> dict:
        """Queries TigerGraph for a multi-hop neighborhood and returns the live sub-graph data payload."""
        start_time = time.time()
        
        # 1. Resolve seed identifier
        identity_resolution = f"Extract the single primary system technical entity id from this query: '{operational_query}'. Return only the raw alphanumeric identifier."
        seed_identity = self.client.models.generate_content(model=self.model_name, contents=identity_resolution).text.strip()
        normalized_seed = re.sub(r'\W+', '_', seed_identity)
        
        # 2. Query your TigerGraph instance using the official vertex signature
        compiled_topology_paths = []
        nodes_payload = []
        edges_payload = []
        tracked_identities = set()
        
        try:
            # Query the multi-hop neighborhood via REST++ endpoints using pyTigerGraph
            raw_neighbors = self.conn.getNeighbors([normalized_seed])
            
            if raw_neighbors:
                # Add seed entity node manually to avoid an empty canvas state tracking window
                nodes_payload.append({"id": normalized_seed, "label": normalized_seed, "title": "Query Seed Entity"})
                tracked_identities.add(normalized_seed)
                
                for edge in raw_neighbors:
                    source_id = edge[0]
                    edge_type = edge[1]
                    target_id = edge[2]
                    
                    path_desc = f"({source_id})--[{edge_type}]-->({target_id})"
                    compiled_topology_paths.append(path_desc)
                    
                    # Track nodes
                    if source_id not in tracked_identities:
                        nodes_payload.append({"id": source_id, "label": source_id, "title": "Source Identity Vertices"})
                        tracked_identities.add(source_id)
                    if target_id not in tracked_identities:
                        nodes_payload.append({"id": target_id, "label": target_id, "title": "Target Identity Vertices"})
                        tracked_identities.add(target_id)
                        
                    # Track structural connection records
                    edges_payload.append({
                        "from": source_id, 
                        "to": target_id, 
                        "label": edge_type,
                        "font": {"size": 10, "color": "#94A3B8"}
                    })
        except Exception as database_error:
            print(f"TigerGraph engine query exception triggered: {str(database_error)}")
            compiled_topology_paths.append(f"Pipeline connectivity warning trace: {str(database_error)}")
            
        # Fallback to keep the UI clean if the dataset isn't fully populated yet
        if not nodes_payload:
            nodes_payload = [{"id": normalized_seed if normalized_seed else "Active_Node", "label": normalized_seed if normalized_seed else "Active_Node"}]
            
        graph_context_string = "\n".join(compiled_topology_paths) if compiled_topology_paths else "No active paths resolved inside graph topology data layers."
        
        # 3. Generate final answer utilizing the true path context
        orchestration_blueprint = f"""
        Synthesize a highly precise explanation for this query using only the validated graph connection contexts.
        Graph Context: {graph_context_string}
        Query: {operational_query}
        """
        response = self.client.models.generate_content(model=self.model_name, contents=orchestration_blueprint)
        
        return {
            "answer": response.text,
            "latency": round(time.time() - start_time, 3),
            "graph_paths_used": graph_context_string,
            "raw_graph_data": {
                "nodes": nodes_payload,
                "edges": edges_payload
            }
        }