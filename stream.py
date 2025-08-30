import mss
from PIL import Image
from io import BytesIO
from flask import Flask, Response
import time

app = Flask(__name__)

@app.route('/stream1')
def stream():
    def generate_frames():
        sct = mss.mss()
        while True:
            # Capture screenshot
            screenshot = sct.grab(sct.monitors[2]) # Adjust monitor index if needed
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

            # Encode to JPEG
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=80) # Adjust quality as needed
            frame_bytes = buffer.getvalue()

            # Yield frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1) # Adjust frame rate

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
