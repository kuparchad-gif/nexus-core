#!/usr/bin/env python3
import http.server
import socketserver
import os
import webbrowser
import threading
import time

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def serve_interface():
    # Change to the directory containing our files
    os.chdir(r'C:\CogniKube-COMPLETE-FINAL')
    
    PORT = 8000
    
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"🌟 Serving Lillith Orb Interface at http://localhost:{PORT}")
        print(f"📁 Serving from: {os.getcwd()}")
        print(f"🎯 Template link: http://localhost:{PORT}/lillith-interface-template.html")
        print("\n🚀 Opening in browser...")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://localhost:{PORT}/lillith-interface-template.html')
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped!")

if __name__ == "__main__":
    serve_interface()