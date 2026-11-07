"""Loopback-only demo with real local inference and no storage of submitted text."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(15)
        return connection, address


def make_server(analyzer, port=8765):
    if not 0 <= port <= 65535:
        raise ValueError('porta inválida')
    lock = threading.Lock()
    page = (Path(__file__).parent / 'web/index.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No submitted text or request paths are logged.

        def reply(self, status, payload, content_type='application/json; charset=utf-8'):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            host = self.headers.get('Host', '')
            allowed_hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            origin = self.headers.get('Origin')
            return host in allowed_hosts and (origin is None or origin in {f'http://{h}' for h in allowed_hosts})

        def do_GET(self):
            if not self.allowed():
                self.reply(403, {'error': 'origem não permitida'})
            elif self.path == '/':
                self.reply(200, page, 'text/html; charset=utf-8')
            elif self.path == '/comparison.json':
                self.reply(200, (Path(__file__).parent / 'web/comparison.json').read_bytes())
            elif self.path == '/healthz':
                self.reply(200, {'ready': True, 'backend': analyzer.backend,
                    'model_version': getattr(getattr(analyzer, 'predictor', None), 'version', 'heuristic-0.1.0'),
                    'policy': analyzer.policy['policy_id'] if analyzer.policy else None})
            else:
                self.reply(404, {'error': 'rota inexistente'})

        def do_POST(self):
            if not self.allowed():
                self.reply(403, {'error': 'origem não permitida'})
                return
            if self.path != '/v1/analyze':
                self.reply(404, {'error': 'rota inexistente'})
                return
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                self.reply(415, {'error': 'envie application/json'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 100000:
                    raise ValueError('body deve ter entre 1 e 100000 bytes')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict) or set(body) != {'text'}:
                    raise ValueError('envie apenas o campo text')
                # Bounded wait instead of queueing concurrent GPU work indefinitely.
                if not lock.acquire(timeout=1):
                    self.reply(503, {'error': 'modelo ocupado; tente novamente'})
                    return
                try:
                    result = analyzer.analyze(body['text'])
                finally:
                    lock.release()
                self.reply(200, result.model_dump(mode='json'))
            except (ValueError, TypeError, UnicodeError) as exc:
                self.reply(400, {'error': str(exc)})
            except (RuntimeError, OSError):
                self.reply(500, {'error': 'falha de inferência local; confira o checkpoint e o dispositivo'})

    return LocalServer(('127.0.0.1', port), Handler)


def serve(analyzer, port):
    server = make_server(analyzer, port)
    print(f'Demo local: http://127.0.0.1:{server.server_port}', file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
