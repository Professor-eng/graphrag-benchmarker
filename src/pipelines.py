import os
import time
import re
import pyTigerGraph as tg
from google import genai
from dotenv import load_dotenv

load_dotenv()

class TigerGraphRAGPipeline:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = "gemini-2.5-flash"
        
        raw_host = os.getenv("TIGERGRAPH_HOST", "")
        clean_host = raw_host.replace("https://", "").replace("http://", "").split("/")[0]
        
        self.conn = tg.TigerGraphConnection(
            host=f"https://{clean_host}",
            graphname="1",
            apiToken=os.getenv("TIGERGRAPH_TOKEN")
        )

    def chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
        """Splits large research papers or documents into overlapping logical segments."""
        chunks = []
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def extract_entities_and_upsert_graph(self, raw_text: str):
        """Processes documents through structured chunk mapping and populates schema elements."""
        chunks = self.chunk_document(raw_text)
        
        for chunk in chunks:
            extraction_prompt = f"""
            Extract primary semantic entities and their explicit structural relations from this text segment.
            Format strictly as unquoted CSV lines matching: SOURCE,RELATION,TARGET
            Do not include markdown headers, formatting, or backticks.
            Text Segment: {chunk}
            """
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=extraction_prompt
                )
                
                lines = response.text.strip().split("\n")
                for line in lines:
                    if not line or "," not in line:
                        continue
                    parts = line.split(",")
                    if len(parts) >= 3:
                        src = parts[0].strip()
                        rel = parts[1].strip()
                        tgt = parts[2].strip()
                        
                        # Direct strongly-typed schema population
                        self.conn.upsertVertex(vertexType="Entity", vertexId=src, attributes={})
                        self.conn.upsertVertex(vertexType="Entity", vertexId=tgt, attributes={})
                        self.conn.upsertEdge(
                            sourceVertexType="Entity", sourceVertexId=src,
                            edgeType="LINKED_TO",
                            targetVertexType="Entity", targetVertexId=tgt,
                            attributes={}
                        )
            except Exception as e:
                print(f"Ingestion anomaly bypassed: {e}")

    def fetch_entire_subgraph(self, seed_term: str, depth: int = 2) -> list:
        """
        Traverses multi-hop connections starting from a target node.
        Uses the corrected pyTigerGraph SDK signature parameters.
        """
        discovered_edges = []
        visited_nodes = set()
        queue = [(seed_term, 0)]
        visited_nodes.add(seed_term)

        while queue:
            current_node, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue
                
            try:
                # Corrected implementation using sourceVertexType and sourceVertexId
                edges = self.conn.getEdges(sourceVertexType="Entity", sourceVertexId=current_node)
                for edge in edges:
                    from_id = edge.get("from_id")
                    to_id = edge.get("to_id")
                    
                    edge_tuple = (from_id, to_id)
                    if edge_tuple not in discovered_edges:
                        discovered_edges.append(edge_tuple)
                        
                    if to_id not in visited_nodes:
                        visited_nodes.add(to_id)
                        queue.append((to_id, current_depth + 1))
            except Exception as e:
                print(f"Traversal edge skip on node '{current_node}': {e}")
                
        return discovered_edges

    def run_graph_rag_pipeline(self, query: str) -> dict:
        """Runs the three-phase TigerGraph GraphRAG workflow."""
        start_time = time.time()
        
        # Phase 1: Entity-Schema Alignment
        alignment_prompt = f"Identify the primary core noun or technical entity keyword from this question. Output ONLY the clean string name: {query}"
        alignment_res = self.client.models.generate_content(model=self.model_name, contents=alignment_prompt)
        seed_term = alignment_res.text.strip().replace("'", "").replace('"', '')
        
        # Phase 2: Multi-Hop Topology Traversal
        edges = self.fetch_entire_subgraph(seed_term, depth=2)
        
        path_strings = []
        graph_json_nodes = set()
        graph_json_edges = []
        
        for src, tgt in edges:
            path_strings.append(f"({src})--[LINKED_TO]->({tgt})")
            graph_json_nodes.add(src)
            graph_json_nodes.add(tgt)
            graph_json_edges.append({"from": src, "to": tgt})
            
        graph_context_string = "\n".join(path_strings) if path_strings else "No direct sub-graphs found."
        
        # Phase 3: Context-Augmented Generation
        contextual_payload = f"""
        You are an advanced intelligence engine. Answer the question using the retrieved knowledge graph pathways.
        
        Question: {query}
        
        Retrieved Sub-Graph Pathways:
        {graph_context_string}
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contextual_payload
        )
        
        return {
            "answer": response.text,
            "latency": round(time.time() - start_time, 3),
            "graph_paths_used": graph_context_string,
            "raw_graph_data": {
                "nodes": [{"id": n, "label": n} for n in graph_json_nodes],
                "edges": graph_json_edges
            }
        }