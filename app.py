from flask import Flask, request, send_file, render_template, jsonify
import importlib
import io
import os
import tempfile
import threading
import uuid

from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

STOCKS = {
    'portra200':    'stocks.portra200',
    'portra800':    'stocks.portra800',
    'ektar100':     'stocks.ektar100',
    'cinestill800t':'stocks.cinestill800t',
    'velvia50':     'stocks.velvia50',
    'hp5_1600':     'stocks.hp5_1600',
}

# job_id → {status, progress, phase, output, preview, stock}
_jobs: dict = {}


def _apply_stock(arr, stock_name):
    """arr: float32 sRGB 0-1. Returns float32 sRGB 0-1."""
    from stocks.base import srgb_to_linear, linear_to_srgb
    path = STOCKS.get(stock_name)
    if not path:
        return arr
    linear = srgb_to_linear(arr)
    result = importlib.import_module(path).process(linear)
    return linear_to_srgb(result)


# ── Photo ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    file  = request.files['image']
    stock = request.form['stock']

    img = Image.open(file).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0

    result     = _apply_stock(arr, stock)
    result_img = Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))

    buf = io.BytesIO()
    result_img.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')


# ── Video ─────────────────────────────────────────────────────────────────────

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    file  = request.files['video']
    stock = request.form.get('stock', 'portra200')

    if stock not in STOCKS:
        return jsonify({'error': f'Unknown stock: {stock}'}), 400

    job_id = uuid.uuid4().hex[:12]

    # Save upload to a temp path — cv2 needs a real file path
    ext = os.path.splitext(file.filename or '')[1] or '.mp4'
    fd_in,  tmp_in  = tempfile.mkstemp(suffix=ext)
    fd_out, tmp_out = tempfile.mkstemp(suffix='.mp4')
    os.close(fd_in)
    os.close(fd_out)
    file.save(tmp_in)

    _jobs[job_id] = {'status': 'queued', 'progress': 0.0, 'phase': 'queued…'}

    def _run():
        try:
            from video.processor import process_video as _process
            _jobs[job_id]['status'] = 'running'

            def _cb(progress, phase):
                _jobs[job_id].update({'progress': progress, 'phase': phase})

            preview_rgb8 = _process(tmp_in, tmp_out, stock, _cb)

            preview_path = None
            if preview_rgb8 is not None:
                import cv2
                preview_path = tmp_out + '_preview.jpg'
                cv2.imwrite(preview_path,
                            cv2.cvtColor(preview_rgb8, cv2.COLOR_RGB2BGR))

            _jobs[job_id] = {
                'status':   'done',
                'progress': 1.0,
                'phase':    'done',
                'output':   tmp_out,
                'preview':  preview_path,
                'stock':    stock,
            }
        except Exception as exc:
            _jobs[job_id] = {'status': 'error', 'error': str(exc), 'progress': 0.0}
        finally:
            try:
                os.unlink(tmp_in)
            except OSError:
                pass

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/video_job/<job_id>')
def video_job(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not_found'}), 404
    # Don't expose raw filesystem paths to the client
    return jsonify({k: v for k, v in job.items() if k not in ('output', 'preview')})


@app.route('/preview/<job_id>')
def preview(job_id):
    job = _jobs.get(job_id)
    if not job or not job.get('preview'):
        return 'No preview', 404
    return send_file(job['preview'], mimetype='image/jpeg')


@app.route('/download_video/<job_id>')
def download_video(job_id):
    job = _jobs.get(job_id)
    if not job or job.get('status') != 'done':
        return 'Not ready', 404
    stock = job.get('stock', 'film')
    return send_file(
        job['output'],
        mimetype='video/mp4',
        as_attachment=True,
        download_name=f'{stock}_film_sim.mp4',
    )


if __name__ == '__main__':
    app.run(debug=True)
