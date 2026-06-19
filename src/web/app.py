"""Flask application entry point for the Petri net web UI.

Usage:
    python -m web.app [--data-dir path] [--port 5000]

Serves two pages:
  /            — Home page listing available configurations
  /petri-net   — Interactive Petri net editor (requires data to exist)
"""

import argparse
import os
import sys
import webbrowser
from threading import Timer

from flask import Flask, render_template

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from web.api import api

PROJECT_ROOT = os.path.dirname(SRC_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def create_app(data_dir: str) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    )
    app.config["DATA_DIR"] = os.path.abspath(data_dir)
    app.register_blueprint(api)

    @app.route("/")
    def home():
        configs = _list_configurations(app.config["DATA_DIR"])
        return render_template("home.html", configs=configs)

    @app.route("/project/<config_name>")
    def project(config_name: str):
        config_dir = os.path.join(app.config["DATA_DIR"], config_name)
        if not os.path.isfile(os.path.join(config_dir, "current.json")):
            return render_template("home.html",
                                   configs=_list_configurations(app.config["DATA_DIR"]),
                                   error=f"Configuration '{config_name}' not found."), 404
        has_original = os.path.isfile(os.path.join(config_dir, "original.json"))
        return render_template("project.html", config_name=config_name,
                               has_original=has_original)

    @app.route("/petri-net/<config_name>")
    def petri_net(config_name: str):
        config_dir = os.path.join(app.config["DATA_DIR"], config_name)
        current = os.path.join(config_dir, "current.json")
        if not os.path.isfile(current):
            return render_template("home.html",
                                   configs=_list_configurations(app.config["DATA_DIR"]),
                                   error=f"Configuration '{config_name}' not found."), 404
        version = __import__("flask").request.args.get("version", "current")
        return render_template("petri_net.html", config_name=config_name,
                               initial_version=version)

    return app


def _list_configurations(data_dir: str) -> list:
    """Scan data_dir for subdirectories containing current.json."""
    configs = []
    if not os.path.isdir(data_dir):
        return configs
    for name in sorted(os.listdir(data_dir)):
        subdir = os.path.join(data_dir, name)
        if os.path.isfile(os.path.join(subdir, "current.json")):
            has_original = os.path.isfile(os.path.join(subdir, "original.json"))
            configs.append({"name": name, "has_original": has_original})
    return configs


def main() -> None:
    """Parse CLI arguments and start the Flask server."""
    argp = argparse.ArgumentParser(description="Petri net web UI")
    argp.add_argument("--port", type=int, default=5000)
    argp.add_argument("--no-browser", action="store_true",
                      help="Do not open browser automatically")
    argp.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                      help=f"Root directory for configurations (default: {DEFAULT_DATA_DIR})")
    args = argp.parse_args()

    app = create_app(args.data_dir)

    if not args.no_browser:
        Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    print(f"Starting server on http://localhost:{args.port}")
    print(f"Data directory: {os.path.abspath(args.data_dir)}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
