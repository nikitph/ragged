#!/usr/bin/env python3
"""
Persistent SentenceTransformer Model Server

Keeps the embedding model loaded in memory for instant vector encoding.
Eliminates the 4+ second model loading time on each search.

Usage:
    # Start the server (run once)
    python model_server.py --start

    # Use in your search script
    python search.py "query" --use-server
"""

import argparse
import json
import numpy as np
import socket
import threading
import time
from sentence_transformers import SentenceTransformer
import pickle
import sys
import os
import signal


class ModelServer:
    """Persistent embedding model server"""

    def __init__(self, model_name="all-MiniLM-L6-v2", host="localhost", port=8888):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.model = None
        self.server_socket = None
        self.running = False

    def load_model(self):
        """Load the embedding model once"""
        print(f"🔄 Loading model: {self.model_name}")
        start_time = time.time()

        self.model = SentenceTransformer(self.model_name)

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        print(f"📏 Vector dimension: {self.model.get_sentence_embedding_dimension()}")

    def start_server(self):
        """Start the model server"""
        self.load_model()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True

            print(f"🚀 Model server running on {self.host}:{self.port}")
            print("💡 Use Ctrl+C to stop the server")
            print("🔥 Model is ready for instant encoding!")

            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket,)
                    )
                    thread.daemon = True
                    thread.start()

                except socket.error:
                    if self.running:
                        print("❌ Socket error occurred")
                    break

        except KeyboardInterrupt:
            print("\n🛑 Shutting down model server...")
        finally:
            self.cleanup()

    def handle_client(self, client_socket):
        """Handle client requests for vector encoding"""
        try:
            # Receive request
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"END_REQUEST" in data:
                    data = data.replace(b"END_REQUEST", b"")
                    break

            if not data:
                return

            # Decode request
            request = pickle.loads(data)
            query = request.get("query", "")

            if not query:
                response = {"error": "No query provided"}
            else:
                # Encode with the loaded model (fast!)
                start_time = time.time()
                vector = self.model.encode([query])[0]
                encode_time = time.time() - start_time

                response = {
                    "vector": vector.tolist(),
                    "dimension": len(vector),
                    "encode_time": encode_time,
                    "model": self.model_name
                }

            # Send response
            response_data = pickle.dumps(response)
            client_socket.sendall(response_data)

        except Exception as e:
            error_response = {"error": str(e)}
            try:
                client_socket.sendall(pickle.dumps(error_response))
            except:
                pass
        finally:
            client_socket.close()

    def cleanup(self):
        """Clean up server resources"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("✅ Model server stopped")


class ModelClient:
    """Client to communicate with the model server"""

    def __init__(self, host="localhost", port=8888):
        self.host = host
        self.port = port

    def encode(self, query):
        """Encode text using the model server"""
        try:
            # Connect to server
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.host, self.port))

            # Send request
            request = {"query": query}
            request_data = pickle.dumps(request) + b"END_REQUEST"
            client_socket.sendall(request_data)

            # Receive response
            response_data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk

            client_socket.close()

            # Decode response
            response = pickle.loads(response_data)

            if "error" in response:
                raise RuntimeError(response["error"])

            return np.array(response["vector"]), response["encode_time"]

        except Exception as e:
            raise RuntimeError(f"Failed to connect to model server: {e}")

    def is_server_running(self):
        """Check if the model server is running"""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(1)
            result = client_socket.connect_ex((self.host, self.port))
            client_socket.close()
            return result == 0
        except:
            return False


def main():
    parser = argparse.ArgumentParser(description="SentenceTransformer Model Server")
    parser.add_argument("--start", action="store_true", help="Start the model server")
    parser.add_argument("--test", action="store_true", help="Test the model server")
    parser.add_argument("--stop", action="store_true", help="Stop the model server")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Model name")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8888, help="Server port")

    args = parser.parse_args()

    if args.start:
        server = ModelServer(args.model, args.host, args.port)

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\n🛑 Received shutdown signal...")
            server.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        server.start_server()

    elif args.test:
        client = ModelClient(args.host, args.port)

        if not client.is_server_running():
            print("❌ Model server is not running!")
            print("Start it with: python model_server.py --start")
            return 1

        # Test encoding
        test_queries = [
            "machine learning algorithms",
            "artificial intelligence",
            "neural networks"
        ]

        print("🧪 Testing model server...")
        total_time = 0

        for query in test_queries:
            try:
                vector, encode_time = client.encode(query)
                total_time += encode_time
                print(f"✅ '{query}' → {len(vector)}D vector in {encode_time:.4f}s")
            except Exception as e:
                print(f"❌ Error encoding '{query}': {e}")
                return 1

        avg_time = total_time / len(test_queries)
        print(f"\n🚀 Average encoding time: {avg_time:.4f}s")
        print("🎯 Server is working perfectly!")

    elif args.stop:
        # For now, just check if running
        client = ModelClient(args.host, args.port)
        if client.is_server_running():
            print("🛑 To stop the server, press Ctrl+C in the server terminal")
        else:
            print("ℹ️  Model server is not running")

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Example integration with search script:
"""
# Modified WikipediaSearchEngine.__init__()

def __init__(self, use_model_server=False, **kwargs):
    if use_model_server:
        self.client = ModelClient()
        if not self.client.is_server_running():
            raise RuntimeError("Model server not running. Start with: python model_server.py --start")
        print("✅ Using model server - instant encoding!")
    else:
        # Load model normally (4+ seconds)
        self.model = SentenceTransformer(model_name)

def search(self, query, **kwargs):
    if hasattr(self, 'client'):
        # Use server - super fast!
        query_vector, encode_time = self.client.encode(query)
        print(f"⚡ Query encoded in {encode_time:.4f}s via server")
    else:
        # Use local model
        query_vector = self.model.encode([query])[0]
"""