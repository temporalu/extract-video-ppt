from json.encoder import INFINITY
import math
import cv2
import os
import shutil
import click
import hashlib
import time
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from .compare import compareImg
from .images2pdf import images2pdf

DEFAULT_PATH = './.extract-video-ppt-tmp-data'
DEFAULT_PDFNAME = 'output.pdf'
DEFAULT_MAXDEGREE = 0.6
CV_CAP_PROP_FRAME_WIDTH = 1920
CV_CAP_PROP_FRAME_HEIGHT = 1080
INFINITY_SIGN = 'INFINITY'
ZERO_SISG = '00:00:00'

VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.m4v', '.mkv', '.avi', '.webm'
}

CACHE_DIR_NAME = 'video2pdt_tmp'
DEBUG = False
SKIP_EXTRACTION = False
CLEANUP_MODE = 'none'
CACHE_BASE_DIR = ''
VIDEO_HASH = ''

URL = ''
OUTPUTPATH = ''
PDFNAME = DEFAULT_PDFNAME
MAXDEGREE = DEFAULT_MAXDEGREE
START_FRAME = 0
END_FRAME = INFINITY
METRIC = 'hist'
MIN_GAP_SEC = 0
CAPTURE_INTERVAL_SEC = 1.0
PICK_MODE = 1

@click.command()
@click.option('--similarity', default = DEFAULT_MAXDEGREE, help = 'The similarity between this frame and the previous frame is less than this value and this frame will be saveed, default: %02g' % (DEFAULT_MAXDEGREE))
@click.option('--pdfname', default = DEFAULT_PDFNAME, help = 'the name of output pdf file, default: video filename or %02s' % (DEFAULT_PDFNAME))
@click.option('--start_frame', default = ZERO_SISG, help = 'start frame time point, default = %02s' % (ZERO_SISG))
@click.option('--end_frame', default = INFINITY_SIGN, help = 'end frame time point, default = %02s' % (INFINITY_SIGN))
@click.option('--outputpath', default = None, help = 'output directory path, default: same directory as video file')
@click.option('--metric', default='hist', type=click.Choice(['hist','ahash','phash','ssim'], case_sensitive=False), help='similarity metric: hist | ahash | phash | ssim')
@click.option('--min_gap', default=0, type=int, help='minimum seconds to skip comparisons after selecting a frame')
@click.option('--interval', default=1.0, type=float, help='seconds between frame captures, default: 1.0')
@click.option('--pick_mode', default=1, type=int, help='1: earliest frame; 2: latest frame')
@click.option('--debug/--no-debug', default=False, help = 'debug mode')
@click.argument('paths', nargs=-1)
def main(
    similarity, pdfname, start_frame, end_frame,
    outputpath, metric, min_gap, interval, pick_mode, debug, paths):
    global URL
    global OUTPUTPATH
    global MAXDEGREE
    global PDFNAME
    global START_FRAME
    global END_FRAME
    global DEBUG
    global METRIC
    global MIN_GAP_SEC
    global CAPTURE_INTERVAL_SEC
    global PICK_MODE

    DEBUG = debug

    input_videos = gather_input_videos(paths)
    if len(input_videos) == 0:
        exitByPrint('no video files found to process')

    if len(input_videos) == 1:
        url = input_videos[0]

        URL = url

        if outputpath is None:
            OUTPUTPATH = extractDirectoryFromPath(url)
        else:
            OUTPUTPATH = outputpath

        MAXDEGREE = similarity
        METRIC = metric.lower()
        MIN_GAP_SEC = max(0, int(min_gap))
        CAPTURE_INTERVAL_SEC = 1.0 if interval is None else max(1e-3, float(interval))
        PICK_MODE = 1 if int(pick_mode) != 2 else 2

        if pdfname == DEFAULT_PDFNAME:
            video_filename = extractFilenameFromPath(url)
            PDFNAME = video_filename + '.pdf'
        else:
            PDFNAME = pdfname

        START_FRAME = hms2second(start_frame)
        END_FRAME = hms2second(end_frame)

        if START_FRAME >= END_FRAME:
            exitByPrint('start >= end can not work')

        prepare()
        start()
        exportPdf()
        clearEnv()
    else:
        run_batch(
            input_videos,
            similarity=similarity,
            pdfname=pdfname,
            start_frame=start_frame,
            end_frame=end_frame,
            outputpath=outputpath,
            metric=metric,
            min_gap=min_gap,
            interval=interval,
            pick_mode=pick_mode,
            debug=debug,
        )

    

def start():
    global CV_CAP_PROP_FRAME_WIDTH
    global CV_CAP_PROP_FRAME_HEIGHT
    global SKIP_EXTRACTION

    vcap = cv2.VideoCapture(URL)
    FPS = int(vcap.get(5))
    TOTAL_FRAME= int(vcap.get(7))

    if TOTAL_FRAME == 0:
        exitByPrint('Please check if the video url is correct')

    CV_CAP_PROP_FRAME_WIDTH = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CV_CAP_PROP_FRAME_HEIGHT = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if START_FRAME > TOTAL_FRAME / FPS:
        exitByPrint('video duration is not support')
    
    if SKIP_EXTRACTION:
        vcap.release()
        cv2.destroyAllWindows()
        return

    use_ffmpeg = shutil.which('ffmpeg') is not None
    if use_ffmpeg:
        start_hms = "%02d:%02d:%02d" % (START_FRAME // 3600, (START_FRAME % 3600) // 60, START_FRAME % 60)
        ffmpeg_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']
        if sys.platform == 'darwin':
            ffmpeg_cmd += ['-hwaccel', 'videotoolbox']
        ffmpeg_cmd += ['-ss', start_hms, '-i', URL]
        if END_FRAME != INFINITY:
            duration = END_FRAME - START_FRAME
            if duration <= 0:
                vcap.release()
                cv2.destroyAllWindows()
                exitByPrint('start >= end can not work')
            ffmpeg_cmd += ['-t', str(duration)]
        fps_value = 1.0 / CAPTURE_INTERVAL_SEC
        ffmpeg_cmd += ['-vf', 'fps=%0.6f' % fps_value, '-threads', str(max(1, (os.cpu_count() or 1))), os.path.join(DEFAULT_PATH, 'frame%05d.jpg')]
        r = subprocess.run(ffmpeg_cmd)
        vcap.release()
        cv2.destroyAllWindows()
        if r.returncode != 0:
            vcap = cv2.VideoCapture(URL)
            vcap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME * FPS)
            frameCount = ((int(TOTAL_FRAME / FPS) if END_FRAME == INFINITY else END_FRAME) - START_FRAME) * FPS
            lastDegree = 0
            lastFrame = []
            readedFrame = 0
            next_capture_frame = CAPTURE_INTERVAL_SEC * FPS
            while(True):
                click.clear()
                print('process:' + str(math.floor(readedFrame / frameCount * 100)) + '%')
                ret, frame = vcap.read()
                if ret:
                    if readedFrame >= frameCount:
                        break
                    readedFrame += 1
                    if readedFrame + 1e-6 < next_capture_frame:
                        continue
                    degree = 0
                    if len(lastFrame):
                        degree = compareImg(frame, lastFrame, METRIC)
                        lastDegree = round(degree, 2)
                    name = DEFAULT_PATH + '/frame' + second2hms(math.ceil((readedFrame + START_FRAME * FPS) / FPS)) + '-' + str(lastDegree) + '-' + ('%05d' % readedFrame) + '.jpg'
                    if not cv2.imwrite(name, frame):
                        exitByPrint('write file failed !')
                    lastFrame = frame
                    next_capture_frame += CAPTURE_INTERVAL_SEC * FPS
                else:
                    break
            vcap.release()
            cv2.destroyAllWindows()
        return
    vcap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME * FPS)
    frameCount = ((int(TOTAL_FRAME / FPS) if END_FRAME == INFINITY else END_FRAME) - START_FRAME) * FPS
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, (os.cpu_count() or 1)))
    lastDegree = 0
    lastFrame = []
    readedFrame = 0
    next_capture_frame = CAPTURE_INTERVAL_SEC * FPS
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 1))) as executor:
        while(True):
            click.clear()
            print('process:' + str(math.floor(readedFrame / frameCount * 100)) + '%')
            ret, frame = vcap.read()
            if ret:
                if readedFrame >= frameCount:
                    break
                readedFrame += 1
                if readedFrame + 1e-6 < next_capture_frame:
                    continue
                degree = 0
                if len(lastFrame):
                    degree = compareImg(frame, lastFrame, METRIC)
                    lastDegree = round(degree, 2)
                name = DEFAULT_PATH + '/frame' + second2hms(math.ceil((readedFrame + START_FRAME * FPS) / FPS)) + '-' + str(lastDegree) + '-' + ('%05d' % readedFrame) + '.jpg'
                futures.append(executor.submit(cv2.imwrite, name, frame))
                lastFrame = frame
                next_capture_frame += CAPTURE_INTERVAL_SEC * FPS
            else:
                break
        for f in futures:
            if not f.result():
                exitByPrint('write file failed !')
    vcap.release()
    cv2.destroyAllWindows()

def prepare():
    global OUTPUTPATH
    global DEFAULT_PATH
    global VIDEO_HASH
    global CACHE_BASE_DIR
    global SKIP_EXTRACTION
    global CLEANUP_MODE
    global DEBUG

    try:
        if not os.path.exists(OUTPUTPATH):
            os.makedirs(OUTPUTPATH)
    except OSError as error:
        exitByPrint(error)

    try:
        VIDEO_HASH = hash_file(URL)
        video_dir = extractDirectoryFromPath(URL)
        CACHE_BASE_DIR = os.path.join(video_dir, CACHE_DIR_NAME)
        interval_str = "%0.6f" % float(CAPTURE_INTERVAL_SEC)
        if os.path.exists(CACHE_BASE_DIR):
            matched = ''
            for name in os.listdir(CACHE_BASE_DIR):
                p = os.path.join(CACHE_BASE_DIR, name)
                if not os.path.isdir(p):
                    continue
                parts = name.split('_')
                if len(parts) == 3:
                    if parts[2] == VIDEO_HASH and parts[1] == interval_str:
                        matched = p
                        break
                elif len(parts) == 2:
                    if parts[1] == VIDEO_HASH and abs(CAPTURE_INTERVAL_SEC - 1.0) < 1e-9:
                        matched = p
                        break
            if matched:
                DEFAULT_PATH = matched
                SKIP_EXTRACTION = True
                CLEANUP_MODE = 'none'
            else:
                ts = time.strftime('%Y%m%d%H%M%S')
                DEFAULT_PATH = os.path.join(CACHE_BASE_DIR, ts + '_' + interval_str + '_' + VIDEO_HASH)
                os.makedirs(DEFAULT_PATH, exist_ok=True)
                SKIP_EXTRACTION = False
                CLEANUP_MODE = 'none' if DEBUG else 'subdir'
        else:
            ts = time.strftime('%Y%m%d%H%M%S')
            DEFAULT_PATH = os.path.join(CACHE_BASE_DIR, ts + '_' + interval_str + '_' + VIDEO_HASH)
            os.makedirs(DEFAULT_PATH, exist_ok=True)
            SKIP_EXTRACTION = False
            CLEANUP_MODE = 'none' if DEBUG else 'subdir'
    except OSError as error:
        exitByPrint(error)

def exportPdf():
    files = []
    for name in os.listdir(DEFAULT_PATH):
        basepath = os.path.join(DEFAULT_PATH, name)
        if not os.path.isfile(basepath):
            continue
        (_, ext) = os.path.splitext(basepath)
        if ext != '.jpg':
            continue
        files.append(basepath)
    files.sort()
    if PICK_MODE == 1:
        selected = []
        last_selected_img = None
        last_selected_sec = None
        for basepath in files:
            filename = os.path.basename(basepath)
            ts = ''
            if 'frame' in filename:
                part = filename.split('frame', 1)[1]
                if '-' in part:
                    ts = part.split('-', 1)[0]
            sec = None
            if ts:
                ss = ts.split('.')
                if len(ss) == 3:
                    sec = int(ss[0]) * 3600 + int(ss[1]) * 60 + int(ss[2])
                    if sec < START_FRAME:
                        continue
                    if END_FRAME != INFINITY and sec > END_FRAME:
                        continue
            img = cv2.imread(basepath)
            if img is None:
                continue
            if len(selected) == 0:
                selected.append(basepath)
                last_selected_img = img
                if sec is not None:
                    last_selected_sec = sec
                continue
            if sec is not None and last_selected_sec is not None:
                if MIN_GAP_SEC > 0 and sec - last_selected_sec < MIN_GAP_SEC:
                    continue
            degree = compareImg(img, last_selected_img, METRIC)
            if degree < MAXDEGREE:
                selected.append(basepath)
                last_selected_img = img
                if sec is not None:
                    last_selected_sec = sec
        pdfPath = os.path.join(DEFAULT_PATH, PDFNAME)
        images2pdf(pdfPath, selected, CV_CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_HEIGHT)
        shutil.copy(pdfPath, OUTPUTPATH)
        print('selected_frames', len(selected), 'total_frames', len(files), 'metric', METRIC, 'threshold', MAXDEGREE, 'min_gap', MIN_GAP_SEC, 'output', os.path.join(OUTPUTPATH, PDFNAME))
    else:
        selected = []
        group_first_img = None
        group_first_sec = None
        group_first_path = None
        group_last_img = None
        group_last_sec = None
        group_last_path = None
        last_selected_sec = None
        for basepath in files:
            filename = os.path.basename(basepath)
            ts = ''
            if 'frame' in filename:
                part = filename.split('frame', 1)[1]
                if '-' in part:
                    ts = part.split('-', 1)[0]
            sec = None
            if ts:
                ss = ts.split('.')
                if len(ss) == 3:
                    sec = int(ss[0]) * 3600 + int(ss[1]) * 60 + int(ss[2])
                    if sec < START_FRAME:
                        continue
                    if END_FRAME != INFINITY and sec > END_FRAME:
                        continue
            img = cv2.imread(basepath)
            if img is None:
                continue
            if group_first_img is None:
                group_first_img = img
                group_first_path = basepath
                group_last_img = img
                group_last_path = basepath
                group_first_sec = sec
                group_last_sec = sec
                continue
            if sec is not None and last_selected_sec is not None:
                if MIN_GAP_SEC > 0 and sec - last_selected_sec < MIN_GAP_SEC:
                    continue
            degree = compareImg(img, group_first_img, METRIC)
            if degree < MAXDEGREE:
                if group_last_path is not None:
                    selected.append(group_last_path)
                    last_selected_sec = group_last_sec
                group_first_img = img
                group_first_path = basepath
                group_last_img = img
                group_last_path = basepath
                group_first_sec = sec
                group_last_sec = sec
            else:
                group_last_img = img
                group_last_path = basepath
                group_last_sec = sec
        if group_last_path is not None:
            selected.append(group_last_path)
        pdfPath = os.path.join(DEFAULT_PATH, PDFNAME)
        images2pdf(pdfPath, selected, CV_CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_HEIGHT)
        shutil.copy(pdfPath, OUTPUTPATH)
        print('selected_frames', len(selected), 'total_frames', len(files), 'metric', METRIC, 'threshold', MAXDEGREE, 'min_gap', MIN_GAP_SEC, 'output', os.path.join(OUTPUTPATH, PDFNAME))

def exitByPrint(str):
    print(str)
    clearEnv()
    exit(1)

def clearEnv():
    global CLEANUP_MODE
    global CACHE_BASE_DIR
    global DEBUG
    if DEBUG:
        return
    if CLEANUP_MODE == 'subdir':
        target = DEFAULT_PATH
    elif CLEANUP_MODE == 'basedir':
        target = CACHE_BASE_DIR
    else:
        target = ''
    if target:
        for _ in range(5):
            try:
                if os.path.exists(target):
                    shutil.rmtree(target)
                break
            except Exception:
                time.sleep(0.2)

def second2hms(second):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return ("%02d.%02d.%02d" % (h, m, s))

def hms2second(hms):
    if hms == INFINITY_SIGN:
        return INFINITY

    h, m, s = hms.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def extractFilenameFromPath(path):
    """Extract filename without extension from a file path"""
    filename = os.path.basename(path)
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext

def extractDirectoryFromPath(path):
    """Extract directory path from a file path"""
    return os.path.dirname(path) or '.'

def is_video_file(path):
    try:
        if not os.path.isfile(path):
            return False
        (_, ext) = os.path.splitext(path)
        return ext.lower() in VIDEO_EXTENSIONS
    except Exception:
        return False

def list_videos_in_dir(path):
    files = []
    try:
        for name in os.listdir(path):
            p = os.path.join(path, name)
            if os.path.isfile(p):
                (_, ext) = os.path.splitext(p)
                if ext.lower() in VIDEO_EXTENSIONS:
                    files.append(p)
    except Exception:
        pass
    return files

def gather_input_videos(paths):
    videos = []
    seen = set()
    for p in paths:
        if not p:
            continue
        x = os.path.expanduser(p)
        if os.path.isdir(x):
            for f in list_videos_in_dir(x):
                if f not in seen:
                    videos.append(f)
                    seen.add(f)
        elif is_video_file(x):
            if x not in seen:
                videos.append(x)
                seen.add(x)
    return videos

def build_args_for_single_video(url, similarity, pdfname, start_frame, end_frame, outputpath, metric, min_gap, interval, pick_mode, debug, force_pdfname):
    args = [sys.executable, '-m', 'video2ppt.video2ppt']
    args += ['--similarity', str(similarity)]
    if force_pdfname and pdfname != DEFAULT_PDFNAME:
        args += ['--pdfname', pdfname]
    args += ['--start_frame', start_frame]
    args += ['--end_frame', end_frame]
    if outputpath is not None:
        args += ['--outputpath', outputpath]
    args += ['--metric', metric]
    args += ['--min_gap', str(int(min_gap))]
    args += ['--interval', str(float(interval))]
    args += ['--pick_mode', str(int(pick_mode))]
    if debug:
        args += ['--debug']
    args += [url]
    return args

def run_batch(video_paths, similarity, pdfname, start_frame, end_frame, outputpath, metric, min_gap, interval, pick_mode, debug):
    max_workers = max(1, min(len(video_paths), (os.cpu_count() or 1)))
    print('batch videos:', len(video_paths), 'workers', max_workers)
    futures = []
    results = []
    # In batch mode, always use per-video default pdf name unless user forces a name and there is only one video
    force_pdfname = (pdfname != DEFAULT_PDFNAME and len(video_paths) == 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for url in video_paths:
            args = build_args_for_single_video(url, similarity, pdfname, start_frame, end_frame, outputpath, metric, min_gap, interval, pick_mode, debug, force_pdfname)
            futures.append(executor.submit(subprocess.run, args, capture_output=False))
        for f in futures:
            r = f.result()
            results.append(r.returncode)
    failed = sum(1 for rc in results if rc != 0)
    if failed:
        print('batch completed with', failed, 'failures out of', len(video_paths))

if __name__ == '__main__':
    main()
