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

            # Extract CSV file and candidate ID
            file_data = None
            candidate_id = None

            for part in msg.iter_parts():
                if part.get_content_disposition() == 'form-data':
                    name = part.get_param('name', header='content-disposition')
                    if name == 'csv':
                        file_data = part.get_payload(decode=True)
                    elif name == 'candidateID':
                        candidate_id = part.get_content().strip()

            if not file_data or not candidate_id:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing file or candidate ID.")
                return

            filename = "uploaded.csv"
            filepath = os.path.join(UPLOAD_DIR, filename)
            with open(filepath, 'wb') as f:
                f.write(file_data)

            print(f"Uploaded to: {filepath}")
            print(f"Candidate ID: {candidate_id}")

            try:
                result = subprocess.run(
                    ['python', 'HR_model.py', filepath, candidate_id],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

                self.send_response(200 if result.returncode == 0 else 500)
                self.send_header("Content-type", "application/json")
                self.end_headers()

                # 🛠 Extract only the final JSON line
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    try:
                        output_json = json.loads(line)
                        self.wfile.write(json.dumps(output_json).encode('utf-8'))
                        return
                    except json.JSONDecodeError:
                        continue

                # If no valid JSON found
                self.wfile.write(b"Failed to parse output.")

            except Exception as e:
                print("Exception occurred:", e)
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Something went wrong during file upload or processing.")
            return

        self.send_response(404)
        self.end_headers()

# Start the server
print(f"✅ Server running at http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), SimpleHTTPRequestHandler) as httpd:
    httpd.serve_forever()
