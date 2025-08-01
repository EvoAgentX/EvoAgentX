import os
import subprocess
import threading
import time
import signal
import atexit
import json
from datetime import datetime
from .api import app
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), 'app.env'), override=True)
target_port = 8001
CREATE_PUBLIC_TUNNEL = False
# TUNNEL_METHOD = "ngrok"
TUNNEL_METHOD = "ssh"
TUNNEL_INFO_PATH = "./server/tunnel_info.json"

tunnel_process = None
ngrok_tunnel = None



def cleanup_tunnel():
    global tunnel_process, ngrok_tunnel
    if TUNNEL_METHOD == "ngrok" and ngrok_tunnel:
        try:
            ngrok_tunnel.close()
            from pyngrok import ngrok
            ngrok.kill()
        except Exception:
            pass
    elif tunnel_process and tunnel_process.poll() is None:
        try:
            tunnel_process.terminate()
            tunnel_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            tunnel_process.kill()
            tunnel_process.wait()
        except Exception:
            pass

def signal_handler(signum, frame):
    print(f"\n🛑 Received signal {signum}, shutting down...")
    cleanup_tunnel()
    exit(0)

def save_tunnel_info(url, port):
    tunnel_info = {
        "public_url": url,
        "local_port": port,
        "local_url": f"http://localhost:{port}",
        "created_at": datetime.now().isoformat(),
        "status": "active"
    }
    
    try:
        with open(TUNNEL_INFO_PATH, "w") as f:
            json.dump(tunnel_info, f, indent=2)
        return True
    except Exception:
        return False

def create_ngrok_tunnel(port):
    global ngrok_tunnel
    try:
        from pyngrok import ngrok
        ngrok_tunnel = ngrok.connect(port)
        public_url = ngrok_tunnel.public_url
        save_tunnel_info(public_url, port)
    except Exception:
        pass

def create_ssh_tunnel(port):
    global tunnel_process
    try:
        tunnel_process = subprocess.Popen(
            ["ssh", "-R", f"80:localhost:{port}", "localhost.run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            text=True
        )
        
        try:
            time.sleep(3)
            
            output_lines = []
            while len(output_lines) < 20:
                try:
                    line = tunnel_process.stdout.readline()
                    if not line:
                        break
                    output_lines.append(line.strip())
                except Exception:
                    break
            
            for line in output_lines:
                if "tunneled with tls termination" in line:
                    import re
                    url_match = re.search(r'https?://[^\s]+\.(localhost\.run|lhr\.life)', line)
                    if url_match:
                        public_url = url_match.group(0)
                        save_tunnel_info(public_url, port)
                        break
        except Exception:
            pass
        
        tunnel_process.wait()
        
    except Exception:
        pass

def create_public_tunnel(port):
    if TUNNEL_METHOD == "ngrok":
        create_ngrok_tunnel(port)
    else:
        create_ssh_tunnel(port)

if __name__ == "__main__":
    import uvicorn
    
    atexit.register(cleanup_tunnel)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    create_tunnel = CREATE_PUBLIC_TUNNEL
    
    if create_tunnel:
        if TUNNEL_METHOD == "ngrok":
            create_public_tunnel(target_port)
        else:
            tunnel_thread = threading.Thread(target=create_public_tunnel, args=(target_port,))
            tunnel_thread.daemon = True
            tunnel_thread.start()
        
        time.sleep(1)
    
    print(f"🖥️  Local server: http://localhost:{target_port}")
    if create_tunnel:
        print("🌍 Public tunnel: URL will be saved to tunnel_info.json")
    else:
        tunnel_info = {
            "public_url": None,
            "local_port": target_port,
            "local_url": f"http://localhost:{target_port}",
            "created_at": datetime.now().isoformat(),
            "status": "inactive"
        }
        with open(TUNNEL_INFO_PATH, "w") as f:
            json.dump(tunnel_info, f, indent=2)
        print("🌍 Public tunnel: Not created")
    

    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=target_port,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server interrupted by user")
    finally:
        # Clean up tunnel
        cleanup_tunnel()


