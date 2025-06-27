import http.server
import os
import socketserver
import subprocess
import json
from email.parser import BytesParser
from email.policy import default

UPLOAD_DIR = "csvfiles"
PORT = 8000

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Domain to model script mapping
DOMAIN_SCRIPTS = {
    "HR": "HR_model.py",
    "Education": "Education_model.py",
    "Banking": "Loan_model.py",
    "Retail": "Retail_model.py",
    "Healthcare": "Healthcare_model.py"
}

class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/upload':
            content_length = int(self.headers.get('Content-Length', 0))
            content_type = self.headers.get('Content-Type', '')
            if 'boundary=' not in content_type:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing boundary in Content-Type.")
                return

            boundary = content_type.split("boundary=")[-1].encode()
            body = self.rfile.read(content_length)
            msg = BytesParser(policy=default).parsebytes(
                b'Content-Type: ' + content_type.encode() + b'\r\n\r\n' + body)

            file_data = None
            input_id = None
            domain = None

            for part in msg.iter_parts():
                if part.get_content_disposition() == 'form-data':
                    name = part.get_param('name', header='content-disposition')
                    if name == 'csv':
                        file_data = part.get_payload(decode=True)
                    elif name == 'id':
                        input_id = part.get_content().strip()
                    elif name == 'domain':
                        domain = part.get_content().strip()

            if not file_data or not input_id or not domain:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing file, ID, or domain.")
                return

            if domain not in DOMAIN_SCRIPTS:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid domain provided.")
                return

            filename = f"uploaded_{domain}.csv"
            filepath = os.path.join(UPLOAD_DIR, filename)
            with open(filepath, 'wb') as f:
                f.write(file_data)

            print(f"Uploaded to: {filepath}")
            print(f"Domain: {domain}, ID: {input_id}")

            try:
                script = DOMAIN_SCRIPTS[domain]
                result = subprocess.run(
                    ['python', script, filepath, input_id],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

                self.send_response(200 if result.returncode == 0 else 500)
                self.send_header("Content-type", "application/json")
                self.end_headers()

                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    try:
                        output_json = json.loads(line)
                        self.wfile.write(json.dumps(output_json).encode('utf-8'))
                        return
                    except json.JSONDecodeError:
                        continue

                self.wfile.write(b"Failed to parse output.")

            except Exception as e:
                print("Exception occurred:", e)
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Internal server error.")
            return

        self.send_response(404)
        self.end_headers()

# Start the server
print(f"✅ Server running at http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), SimpleHTTPRequestHandler) as httpd:
    httpd.serve_forever()
