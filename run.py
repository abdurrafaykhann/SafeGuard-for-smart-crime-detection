"""Entry point.  Dev:  python run.py      Prod:  waitress is used automatically unless --dev."""
import argparse

from safeguard import create_app

app = create_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--dev", action="store_true", help="Flask debug server with auto-reload")
    args = parser.parse_args()
    if args.dev:
        app.run(args.host, args.port, debug=True, use_reloader=False, threaded=True)
    else:
        from waitress import serve
        # each open MJPEG / SSE connection occupies a thread, so keep the pool generous
        serve(app, host=args.host, port=args.port, threads=48)
