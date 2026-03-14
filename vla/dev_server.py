#!/usr/bin/env python3
"""
Development Server with Hot Reload

Auto-restarts the SmolVLA server when source files change.
Useful for rapid iteration during development.

Usage: python dev_server.py
"""

import subprocess
import sys
import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [DEV] %(message)s"
)
logger = logging.getLogger(__name__)

SERVER_SCRIPT = Path(__file__).parent / "vla_production_server.py"
WATCH_PATTERNS = [".py"]  # Watch Python files
IGNORE_PATTERNS = ["__pycache__", ".pyc", ".git"]

class ServerRestartHandler(FileSystemEventHandler):
    """Restart server when watched files change."""
    
    def __init__(self, process_manager):
        self.process_manager = process_manager
        self.last_restart = time.time()
        self.restart_cooldown = 2.0  # Cooldown to avoid rapid restarts
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        
        # Ignore unwanted patterns
        if any(p in str(filepath) for p in IGNORE_PATTERNS):
            return
        
        # Only watch Python files in vla/ directory
        if filepath.suffix in WATCH_PATTERNS and "vla" in str(filepath):
            now = time.time()
            if now - self.last_restart > self.restart_cooldown:
                logger.info(f"File changed: {filepath.name}")
                self.process_manager.restart()
                self.last_restart = now


class ServerProcessManager:
    """Manage server process lifecycle."""
    
    def __init__(self, script_path):
        self.script_path = script_path
        self.process = None
    
    def start(self):
        """Start the server process."""
        logger.info(f"Starting server: {self.script_path}")
        self.process = subprocess.Popen(
            [sys.executable, str(self.script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Log output in a thread
        import threading
        def log_output():
            for line in self.process.stdout:
                print(line.rstrip())
        
        thread = threading.Thread(target=log_output, daemon=True)
        thread.start()
        
        # Wait for startup
        time.sleep(3)
        logger.info("✓ Server started")
    
    def stop(self):
        """Stop the server process."""
        if self.process and self.process.poll() is None:
            logger.info("Stopping server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, killing...")
                self.process.kill()
            logger.info("✓ Server stopped")
    
    def restart(self):
        """Restart the server."""
        logger.info("Restarting server due to file changes...")
        self.stop()
        time.sleep(1)
        self.start()


def main():
    """Run dev server with hot reload."""
    logger.info("=" * 70)
    logger.info("SmolVLA Development Server with Hot Reload")
    logger.info("=" * 70)
    
    manager = ServerProcessManager(SERVER_SCRIPT)
    
    try:
        # Start initial server
        manager.start()
        
        # Setup file watcher
        handler = ServerRestartHandler(manager)
        observer = Observer()
        observer.schedule(handler, path=str(SERVER_SCRIPT.parent), recursive=False)
        observer.start()
        
        logger.info("✓ File watcher started")
        logger.info("Watching for changes in vla/ directory... (Press Ctrl+C to stop)")
        logger.info("=" * 70)
        
        # Keep running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        observer.stop()
        observer.join()
        manager.stop()
        logger.info("✓ Development server stopped")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        manager.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
