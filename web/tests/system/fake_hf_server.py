"""A tiny fake Hugging Face server, for the system tests (T07).

It answers just enough of the Hub protocol for `huggingface_hub.snapshot_download` (the call the web app makes at
start-up): the repo listing and the file downloads. The files it serves are a few bytes of fake data, NOT real weights.
It requires the token `FAKE_TOKEN`, so the test also proves the app sends its token to the endpoint it is told to use.
"""
import hashlib
import http.server
import json
import threading

FAKE_TOKEN = 'hf_FAKEtokenFAKEtokenFAKEtoken0123456789'
WEIGHTS_REPO = 'huy-le/checking-driver-weights'
SHA = 'a' * 40
FILES = {
    'final_1d.pth': b'fake weights, not a torch file',
    'kine_norm_stats.json': b'{"mean": [0, 0, 0], "std": [1, 1, 1]}',
    'video_3d.pth': b'fake video weights',
    'fusion.pth': b'fake fusion weights',
}


class FakeHuggingFace:
    def __init__(self, port):
        self.requests = []                       # (method, path, authorised?)
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self):
                ok = self.headers.get('Authorization') == f'Bearer {FAKE_TOKEN}'
                outer.requests.append((self.command, self.path, ok))
                if not ok:
                    return self._send(401, b'{"error": "Invalid credentials"}', 'application/json')
                if self.path == f'/api/models/{WEIGHTS_REPO}/revision/main':
                    body = json.dumps({'id': WEIGHTS_REPO, 'sha': SHA,
                                       'siblings': [{'rfilename': f} for f in FILES]}).encode()
                    return self._send(200, body, 'application/json')
                if self.path.startswith(f'/api/models/{WEIGHTS_REPO}/tree/{SHA}'):
                    body = json.dumps([{'type': 'file', 'path': f, 'size': len(d), 'oid': hashlib.sha1(d).hexdigest()}
                                       for f, d in FILES.items()]).encode()
                    return self._send(200, body, 'application/json')
                prefix = f'/{WEIGHTS_REPO}/resolve/{SHA}/'
                if self.path.startswith(prefix) and self.path[len(prefix):] in FILES:
                    return self._send(200, FILES[self.path[len(prefix):]], 'application/octet-stream', file=True)
                return self._send(404, b'{"error": "Not found"}', 'application/json')

            def _send(self, code, body, ctype, file=False):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                if file:
                    self.send_header('ETag', '"' + hashlib.sha1(body).hexdigest() + '"')
                    self.send_header('X-Repo-Commit', SHA)
                    self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()
                if self.command != 'HEAD':
                    self.wfile.write(body)

            do_GET = do_HEAD = _answer

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(('0.0.0.0', port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def downloads(self):
        """How many file downloads (GET on /resolve/) the app made."""
        return sum(1 for m, p, ok in self.requests if m == 'GET' and '/resolve/' in p and ok)
