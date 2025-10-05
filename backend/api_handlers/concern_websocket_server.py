"""
Real-time WebSocket server for CONCERN EWS streaming
Provides instant updates to frontend clients
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Set, Optional
import websockets
from websockets.server import WebSocketServerProtocol
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    from .advanced_realtime_concern_ews import get_advanced_realtime_concern_ews
except ImportError:
    from api_handlers.advanced_realtime_concern_ews import get_advanced_realtime_concern_ews

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConcernEWSWebSocketServer:
    """WebSocket server for real-time CONCERN EWS updates"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.server = None
        self.running = False
        self.ews_engine = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize EWS engine in background
        threading.Thread(target=self._init_ews_engine, daemon=True).start()
    
    def _init_ews_engine(self):
        """Initialize EWS engine in background thread"""
        try:
            self.ews_engine = get_advanced_realtime_concern_ews()
            logger.info("✅ EWS engine initialized for WebSocket server")
        except Exception as e:
            logger.error(f"❌ Failed to initialize EWS engine: {e}")
    
    async def register_client(self, websocket: WebSocketServerProtocol, patient_id: str):
        """Register a client for patient updates"""
        if patient_id not in self.clients:
            self.clients[patient_id] = set()
        
        self.clients[patient_id].add(websocket)
        logger.info(f"📱 Client registered for patient {patient_id} (Total: {len(self.clients[patient_id])})")
    
    async def unregister_client(self, websocket: WebSocketServerProtocol, patient_id: str):
        """Unregister a client"""
        if patient_id in self.clients:
            self.clients[patient_id].discard(websocket)
            if not self.clients[patient_id]:
                del self.clients[patient_id]
            logger.info(f"📱 Client unregistered for patient {patient_id}")
    
    async def broadcast_to_patient_clients(self, patient_id: str, data: Dict):
        """Broadcast data to all clients monitoring a specific patient"""
        if patient_id not in self.clients:
            return
        
        # Remove disconnected clients
        connected_clients = set()
        for client in self.clients[patient_id].copy():
            try:
                await client.ping()
                connected_clients.add(client)
            except:
                self.clients[patient_id].discard(client)
        
        self.clients[patient_id] = connected_clients
        
        # Broadcast to connected clients
        if connected_clients:
            message = json.dumps(data)
            disconnected = []
            
            for client in connected_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(client)
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    disconnected.append(client)
            
            # Clean up disconnected clients
            for client in disconnected:
                self.clients[patient_id].discard(client)
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle individual client connections"""
        patient_id = None
        try:
            # Parse path to get patient ID
            # Expected format: /concern/{patient_id}
            path_parts = path.strip('/').split('/')
            if len(path_parts) >= 2 and path_parts[0] == 'concern':
                patient_id = path_parts[1]
            else:
                await websocket.send(json.dumps({
                    'error': 'Invalid path format. Use /concern/{patient_id}'
                }))
                return
            
            # Register client
            await self.register_client(websocket, patient_id)
            
            # Send initial data
            if self.ews_engine:
                initial_data = await self._get_patient_concern_data(patient_id)
                if initial_data:
                    await websocket.send(json.dumps({
                        'type': 'initial_data',
                        'patient_id': patient_id,
                        'data': initial_data,
                        'timestamp': datetime.now().isoformat()
                    }))
            
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, patient_id, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'error': 'Invalid JSON format'
                    }))
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
                    await websocket.send(json.dumps({
                        'error': 'Internal server error'
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected for patient {patient_id}")
        except Exception as e:
            logger.error(f"❌ Error handling client: {e}")
        finally:
            if patient_id:
                await self.unregister_client(websocket, patient_id)
    
    async def _handle_client_message(self, websocket: WebSocketServerProtocol, 
                                   patient_id: str, data: Dict):
        """Handle messages from clients"""
        try:
            message_type = data.get('type')
            
            if message_type == 'ping':
                await websocket.send(json.dumps({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }))
            
            elif message_type == 'refresh':
                # Force refresh of patient data
                if self.ews_engine:
                    concern_data = await self._get_patient_concern_data(patient_id, force_refresh=True)
                    await websocket.send(json.dumps({
                        'type': 'refresh_response',
                        'patient_id': patient_id,
                        'data': concern_data,
                        'timestamp': datetime.now().isoformat()
                    }))
            
            elif message_type == 'subscribe_alerts':
                # Subscribe to critical alerts for all patients
                await websocket.send(json.dumps({
                    'type': 'alert_subscription_confirmed',
                    'timestamp': datetime.now().isoformat()
                }))
            
            else:
                await websocket.send(json.dumps({
                    'error': f'Unknown message type: {message_type}'
                }))
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await websocket.send(json.dumps({
                'error': 'Failed to process message'
            }))
    
    async def _get_patient_concern_data(self, patient_id: str, force_refresh: bool = False) -> Optional[Dict]:
        """Get patient concern data asynchronously"""
        if not self.ews_engine:
            return None
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            concern_data = await loop.run_in_executor(
                self.executor, 
                self.ews_engine.get_realtime_concern_stream, 
                patient_id
            )
            return concern_data
        except Exception as e:
            logger.error(f"Error getting concern data for {patient_id}: {e}")
            return None
    
    async def start_streaming_loop(self):
        """Start continuous streaming of updates to clients"""
        logger.info("🔄 Starting streaming loop")
        
        while self.running:
            try:
                # Get all monitored patients
                all_patient_ids = set()
                for patient_id in self.clients.keys():
                    all_patient_ids.add(patient_id)
                
                # Stream updates for each patient with active clients
                for patient_id in all_patient_ids:
                    if patient_id in self.clients and self.clients[patient_id]:
                        concern_data = await self._get_patient_concern_data(patient_id)
                        
                        if concern_data:
                            # Add real-time metadata
                            stream_message = {
                                'type': 'realtime_update',
                                'patient_id': patient_id,
                                'data': concern_data,
                                'server_timestamp': datetime.now().isoformat(),
                                'stream_sequence': int(time.time() * 1000)  # Millisecond timestamp
                            }
                            
                            # Broadcast to all clients for this patient
                            await self.broadcast_to_patient_clients(patient_id, stream_message)
                
                # Wait before next update cycle (reduced frequency)
                
            except Exception as e:
                logger.error(f"❌ Error in streaming loop: {e}")
                continue
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"🚀 Starting CONCERN EWS WebSocket server on {self.host}:{self.port}")
        
        try:
            self.running = True
            
            # Start the WebSocket server
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10,
                max_size=1024*1024,  # 1MB max message size
                compression=None
            )
            
            # Start streaming loop
            streaming_task = asyncio.create_task(self.start_streaming_loop())
            
            logger.info(f"✅ WebSocket server started successfully on ws://{self.host}:{self.port}")
            logger.info("📡 Real-time streaming active")
            
            # Wait for server to be stopped
            await self.server.wait_closed()
            
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
            raise
        finally:
            self.running = False
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        logger.info("🛑 Stopping WebSocket server")
        self.running = False
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all client connections
        for patient_clients in self.clients.values():
            for client in patient_clients:
                try:
                    await client.close()
                except:
                    pass
        
        self.clients.clear()
        logger.info("✅ WebSocket server stopped")

# Global server instance
_websocket_server = None

def get_concern_websocket_server() -> ConcernEWSWebSocketServer:
    """Get the global WebSocket server instance"""
    global _websocket_server
    
    if _websocket_server is None:
        _websocket_server = ConcernEWSWebSocketServer()
    
    return _websocket_server

async def start_concern_websocket_server(host: str = "localhost", port: int = 8765):
    """Start the CONCERN WebSocket server"""
    server = ConcernEWSWebSocketServer(host, port)
    await server.start_server()

if __name__ == "__main__":
    # Run the WebSocket server
    import argparse
    
    parser = argparse.ArgumentParser(description='CONCERN EWS WebSocket Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    
    args = parser.parse_args()
    
    try:
        asyncio.run(start_concern_websocket_server(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
