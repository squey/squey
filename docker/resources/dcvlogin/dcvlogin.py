#!/usr/bin/env python3

################################################################################
# MIT License
#
# © ESI Group, 2015
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

__author__ = "Jean-Baptiste Leonesio <jean-baptiste@leonesio.fr>"
__copyright__ = "Copyright 2021, Jean-Baptiste Leonesio <jean-baptiste@leonesio.fr>"

from http.server import SimpleHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import pam
import sys
import os
import ssl
import cgi
import secrets
import subprocess
import json
import datetime

PORT = 8445

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'web_root'))

class Handler(SimpleHTTPRequestHandler):
    
    def translate_path(self, path):
        if path == "/":
            return "index.html"
        path = SimpleHTTPRequestHandler.translate_path(self, path)
        relpath = os.path.relpath(path, os.getcwd())
        try:
            with open(relpath):
                return relpath
        except IOError:
            return '/usr/share/dcv/www/' + relpath
            
    def do_POST(self):
        # Parse the form data posted
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type']})
        
        user=form.getvalue("login")
        password=form.getvalue("password")

        if not pam.pam().authenticate(user, password, "auth"):
            self.send_response(401)
            self.end_headers()
            return
            
        # Create a DCV virtual session
        session_id = user
        sessions_json = subprocess.check_output(['/usr/bin/dcv', 'list-sessions', '-j'])
        sessions = json.loads(sessions_json)
        while True:
            try:
                if not user in [s['id'] for s in sessions]:
                    subprocess.check_output(['/usr/bin/dcv', 'create-session', '--user={}'.format(user), '--owner={}'.format(user), session_id], stderr=subprocess.STDOUT)
                break
            except subprocess.CalledProcessError as e:
                error_msg = e.output.decode()
                if error_msg.find('All licenses in use') != -1: # No remaining DCV license token : close oldest idle session if any
                    idle_sessions = [{'id': s['id'], 'timestamp': s['last-disconnection-time']} for s in sessions if s['num-of-connections'] == 0]
                    idle_sessions.sort(key=lambda s: datetime.datetime.strptime(s['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ"))
                    if len(idle_sessions) > 0:  # Close oldest idle session and remove associated auth token(s)
                        idle_session_id=idle_sessions[0]['id']
                        subprocess.call(['/usr/bin/dcv', 'close-session', idle_session_id])
                        subprocess.run(['/usr/bin/dcvsimpleextauth', 'remove-auth', '--session={}'.format(idle_session_id)])
                    else: # No idle session to close
                        self.send_response(403)
                        self.end_headers()
                        return

        # Add authentification token to virtual session
        token=secrets.token_hex()
        subprocess.run(['/usr/bin/dcvsimpleextauth', 'add-user', '--append', '--user={}'.format(user), '--session={}'.format(session_id)], input=token, encoding='ascii')
        
        self.send_response(200)
        self.end_headers()
        host = self.headers.get('Host')
        self.wfile.write(bytes('https://{}/dcv/?authToken={}#{}'.format(host, token, session_id), 'utf-8'))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

print('Server listening on port ' + str(PORT) + '...')
httpd = ThreadedHTTPServer(("", PORT), Handler)
httpd.socket = ssl.wrap_socket (
    httpd.socket,
    keyfile='/etc/dcv/dcv.key',
    certfile='/etc/dcv/dcv.pem',
    server_side=True
)
httpd.serve_forever()
