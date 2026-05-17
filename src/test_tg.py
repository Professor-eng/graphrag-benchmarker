import os
from dotenv import load_dotenv
import pyTigerGraph as tg

load_dotenv()

raw_host = os.getenv("TIGERGRAPH_HOST", "http://127.0.0.1")
clean_host = raw_host.replace("https://", "").replace("http://", "").split("/")[0]

conn = tg.TigerGraphConnection(
    host=f"http://{clean_host}" if "127.0.0.1" in clean_host else f"https://{clean_host}",
    graphname="CyberSecurity_Network_Graph",
    username=os.getenv("TG_USERNAME", "tigergraph"),
    password=os.getenv("TG_PASSWORD", "tigergraph"),
    apiToken=os.getenv("TIGERGRAPH_TOKEN")
)

print("Pinging TigerGraph cluster instance status...")
print("Heartbeat response:", conn.ping())
print("Vertices currently tracked on schema context layout:", conn.getVertexTypes())