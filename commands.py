import numpy as np
import io
import os
import cv2 as cv

############################################################################################
from contextlib import contextmanager

@contextmanager
def video_capture_wrapper(*args, **kwargs):
    try:
        vid_stream = cv.VideoCapture(*args, **kwargs)
        yield vid_stream
    finally:
         vid_stream.release()

############################################################################################

from dlt import dlt, transform
from draw_marking import draw_transformed_grid

def draw_markings(median_frame, T, T_inv):
    draw_transformed_grid(median_frame, T, T_inv)
    return median_frame

############################################################################################

import sqlite3

db_path = "traffic_watcher.db"

def open_database():
    # make sqlite work with np.array
    # https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
    def adapt_array(arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)

    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("ARRAY", convert_array)

    db_connection = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) # connect to DB in file
    db_connection.row_factory = sqlite3.Row # use special class for rows

    db_cursor = db_connection.cursor() # create cursor object

    # turn on foreign keys that are needed for relations
    db_cursor.execute("PRAGMA foreign_keys = ON;")
    
    return db_connection, db_cursor

def close_database(db_connection, db_cursor):
    db_connection.commit()
    db_connection.close()

############################################################################################

def new_session( namespace ):
    # lab_2.py new_session <list of .mp4's>
    # -> creates new session for detection (overwrites old)
    # gets list of files, checks for availability,
    # writes them into DB and exits;
    # also creates median_frame for later use and store to DB
    
    video_files_paths = namespace.files # file paths from sys.argv

    # check for availability
    print("Checking files")
    for path in video_files_paths:
        # check if file exists
        if not os.path.isfile(path):
            print("Error opening {}".format(path))
            os._exit(os.EX_OK)
        # try to open as video
        with video_capture_wrapper(path) as vid:
            if not vid.isOpened():
               print("Error opening {}".format(path))
               os._exit(os.EX_OK)

    # resetting database
    print("Resetting database")
    db_connection, db_cursor = open_database()

    db_cursor.execute("DROP TABLE IF EXISTS file_list")
    db_cursor.execute("CREATE TABLE file_list (path TEXT NOT NULL)")

    db_cursor.execute("DROP TABLE IF EXISTS transform_matrix")
    db_cursor.execute("CREATE TABLE transform_matrix (id INTEGER, matrix ARRAY)")
    
    db_cursor.execute("DROP TABLE IF EXISTS mean_fps")
    db_cursor.execute("CREATE TABLE mean_fps (fps REAL)")

    db_cursor.execute("DROP TABLE IF EXISTS median_image")
    db_cursor.execute("CREATE TABLE median_image (id INTEGER, image ARRAY)")

    db_cursor.execute("DROP TABLE IF EXISTS lines")
    db_cursor.execute('''
            CREATE TABLE lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                color_r INTEGER NOT NULL,
                color_g INTEGER NOT NULL,
                color_b INTEGER NOT NULL,
                knots ARRAY 
            )
    ''')

    db_cursor.execute("DROP TABLE IF EXISTS detectors")
    db_cursor.execute('''
            CREATE TABLE detectors (
                number_on_line INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                w INTEGER NOT NULL,
                h INTEGER NOT NULL,
                median_frame_color REAL,
                mean_color ARRAY,
                binarized ARRAY,
                detections ARRAY,
                line_id INTEGER NOT NULL,
                FOREIGN KEY (line_id) REFERENCES lines(id) ON DELETE CASCADE
            )
    ''') 
    
    print("Adding files to database")
    db_cursor.executemany("INSERT INTO file_list VALUES (?)", 
            [(p,) for p in video_files_paths])

    print("Computing median frame")

    # get frames count in all videos
    frames_count = []
    for path in video_files_paths:
        with video_capture_wrapper(path) as vid:
            frames_count.append( # add to list
                    vid.get(cv.CAP_PROP_FRAME_COUNT) # frame count
                    )
    frames_to_median = 21 #201 # may be too many
    total_frames = sum( frames_count )
    frames = [] # list of frames
    # get frames
    for path, fr_count in zip(video_files_paths, frames_count):
        with video_capture_wrapper(path) as vid:
            frame_numbers = (
                    fr_count
                    * np.random.uniform( size = int( frames_to_median * fr_count/total_frames ) )
                    )
            for i in frame_numbers:
                vid.set(cv.CAP_PROP_POS_FRAMES, int(i))
                ret, frame = vid.read()
                frames.append(frame)
    
    # calculate the median along the time axis
    median_frame = np.median(frames, axis = 0).astype(dtype = np.uint8)
    
    # set default transform_matrix 
    T = np.diag([median_frame.shape[1], median_frame.shape[0], 1])
    T_inv = np.diag([1/median_frame.shape[1], 1/median_frame.shape[0], 1])
    db_cursor.executemany("INSERT INTO transform_matrix VALUES (?, ?)",
            [(1, T), (-1, T_inv)] )

    # adding median frame to database
    print("Adding median frame to database")
    db_cursor.executemany("INSERT INTO median_image VALUES (?, ?)",
            [(1, median_frame), (2, median_frame), 
                (3, draw_markings(np.copy(median_frame), T, T_inv))]) 
    # 1 -> median frame
    # 2 -> median frame with lines and detectors
    # 3 -> median frame with markings 

    # count mean fps for session
    frames_count = 0
    duration = 0
    for path in video_files_paths:
        with video_capture_wrapper(path) as vid:
            frames_count = frames_count + vid.get(cv.CAP_PROP_FRAME_COUNT)
            duration = duration + vid.get(cv.CAP_PROP_FRAME_COUNT) / vid.get(cv.CAP_PROP_FPS)
    mean_fps = frames_count / duration

    # add mean fps to database
    db_cursor.execute("INSERT INTO mean_fps VALUES (?)", (mean_fps,) )

    # save and close database
    close_database(db_connection, db_cursor)

    print("Displaying median frame")
    # make window
    cv.namedWindow("Median frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("Median frame", 800, 600) # set window size
    # display median frame
    print("Press ESC to close window")
    while True:
        cv.imshow("Median frame", median_frame)
        if cv.waitKey(0) & 0xFF == 27: break # until ESC is pressed
    # close windows
    cv.destroyAllWindows()

############################################################################################

def print_line_info(line_id, colorRGB, knots, detectors_count):
    print("\t id = {}, color = ({}, {}, {}), {} knots, {} detectors".format(
        line_id, colorRGB[0], colorRGB[1], colorRGB[2], knots.shape[0], detectors_count))

def show_detectors( namespace ):
    # lab_2.py show_detectors
    # -> shows median_frame with lines and detectors
    # shows line and detector numbers
    # exits on ESC
    db_connection, db_cursor = open_database()

    print("File list:")
    db_cursor.execute("SELECT * FROM file_list")
    for row in db_cursor.fetchall():
        print("\t{}".format(row[0]))

    # fetch median image from database
    db_cursor.execute("SELECT image FROM median_image WHERE id = 2")
    median_frame = db_cursor.fetchone()[0] 

    print("Lines:")
    db_cursor.execute("SELECT * FROM lines")
    rows = db_cursor.fetchall()
    if not rows:
        print("No lines in database")
    for row in rows:
        line_id = row[0]
        db_cursor.execute("SELECT COUNT(*) FROM detectors WHERE line_id = ?", (line_id,))
        detectors_count = db_cursor.fetchone()[0]
        print_line_info(line_id, (row[1], row[2], row[3]), row[4], detectors_count)
        
    # save and close database
    close_database(db_connection, db_cursor)

    # make window
    cv.namedWindow("Median frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("Median frame", 800, 600) # set window size
    # display median frame
    while True:
        cv.imshow("Median frame", median_frame)
        if cv.waitKey(0) & 0xFF == 27: break # until ESC is pressed
    # close windows
    cv.destroyAllWindows()

############################################################################################

from scipy.interpolate import CubicSpline

def draw_knot(median_frame, x, y, colorBGR):
    cv.circle(median_frame, (x, y), 3, colorBGR, -1, cv.LINE_AA)

def draw_line(median_frame, knots, colorBGR):
    for (x, y) in knots: # redraw knots
        draw_knot(median_frame, x, y, colorBGR)
    if knots.shape[0] > 1: # if 2 or more knots
        # interpolate line
        t = np.linspace(0, 1, knots.shape[0])
        line_spline = CubicSpline(t, knots)
        t = np.linspace(0, 1, 10000)
        for ti in t:
            r = line_spline(ti)
            # color pixel
            median_frame[ int(r[1]), int(r[0]) ] = colorBGR
    else: # only 1 knot
        # color single pixel
        median_frame[ int(knots[0,0]), int(knots[0,1]) ] = colorBGR

def draw_detector(median_frame, detector_number, detector_coords, detector_size, colorBGR):
    cv.rectangle( median_frame, # place a rectangle 
            ( int(detector_coords[0] - detector_size[0]/2), int(detector_coords[1] - detector_size[1]/2) ),
            ( int(detector_coords[0] + detector_size[0]/2), int(detector_coords[1] + detector_size[1]/2) ),
            colorBGR,  2, cv.LINE_AA)
    if detector_number is not None:
        cv.putText( median_frame, "{}".format(detector_number), # draw detector number
                 ( int(detector_coords[0] - detector_size[0]/3), int(detector_coords[1] + detector_size[1]/4) ),
                 cv.FONT_HERSHEY_SIMPLEX, 0.7, colorBGR, 1, cv.LINE_AA)

def place_line( namespace ):
    # lab_2.py place_line --colorRGB <R> <G> <B> --detector_size <W> <H>
    # -> shows median_frame with lines and detectors
    # allow to place knots on image
    # after ESC interpolates line and places detectors
    # exits on second ESC
    db_connection, db_cursor = open_database()
    
    # fetch median image from database
    db_cursor.execute("SELECT image FROM median_image WHERE id = 2")
    median_frame = db_cursor.fetchone()[0] 
 
    # choose new line a color unless preset
    if namespace.colorRGB is None:
        # get count of existing lines
        db_cursor.execute("SELECT COUNT(id) FROM lines")
        lines_count = db_cursor.fetchone()[0]
        if lines_count >= 16:
            colorRGB = (0, 255, 0) # lime
        else:
            preset_line_colors = [
                (0, 255, 255), # aqua	
                (0, 0, 0), # black
                (0, 0, 255),	# blue
                (255, 0, 255), # fuchsia
                (128, 128, 128),	# gray
                (0, 128, 0),	# green
                (0, 255, 0),	# lime
                (128, 0, 0), # maroon
                (0, 0, 128),	# navy
                (128, 128, 0), # olive
                (128, 0, 128), # purple
                (255, 0, 0), # red
                (192, 192, 192),	# silver
                (0, 128, 128), # teal
                (255, 255, 255),	# white
                (255, 255, 0) # yellow
            ] 
            colorRGB = preset_line_colors[lines_count]
    else:
        colorRGB = namespace.colorRGB
    colorBGR = (colorRGB[2], colorRGB[1], colorRGB[0])
    
    detector_size = namespace.detector_size

    knots = [] # list of lines knots

    # make window
    cv.namedWindow("Median frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("Median frame", 800, 600) # set window size

    # mouse callback function
    def add_knot(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            knots.append( (x, y) ) # add to list
            draw_knot(median_frame, x, y, colorBGR)
            cv.imshow("Median frame", median_frame)

    # add mouse callback to window
    cv.setMouseCallback("Median frame", add_knot)

    print("Click on image to place a line knot")
    print("Press ESC to finish")

    # display median frame
    while True:
        cv.imshow("Median frame", median_frame)
        if cv.waitKey(0) & 0xFF == 27: break # until ESC is pressed

    # transform list to array
    knots = np.array(knots)

    if knots.shape[0] > 1: # if 2 or more knots
        # interpolate line
        t = np.linspace(0, 1, knots.shape[0])
        line_spline = CubicSpline(t, knots)
        t = np.linspace(0, 1, 10000)

        # unset mouse callback
        cv.setMouseCallback("Median frame", lambda event, x, y, flags, parameters : None)

        # place detectors
        detector_coords = []
        r_prev = np.array( [np.Inf, np.Inf] )
        for ti in t:
            r = line_spline(ti)
            dr = np.abs( r - r_prev )
            if dr[0] >= detector_size[0] or dr[1] >= detector_size[1]: # if far enough
                # place another detector
                detector_coords.append( (int(r[0]), int(r[1]) ) )
                r_prev = r

    elif knots.shape[0] == 1:
        # place single detector
        detector_coords = [ ( int(knots[0,0]), int(knots[0,1]) ) ]
    else: # no knots
        # just quit
        # save and close database
        close_database(db_connection, db_cursor)
        # close windows
        cv.destroyAllWindows()
        os._exit(os.EX_OK)
   
    draw_line(median_frame, knots, colorBGR)
    for n, dr in enumerate(detector_coords):
        draw_detector(median_frame, n, dr, detector_size, colorBGR)

    print("Saving line and detectors to database")
    # add line to database
    db_cursor.execute("INSERT INTO lines (color_r, color_g, color_b, knots) VALUES (?, ?, ?, ?)",
            (colorRGB[0], colorRGB[1], colorRGB[2], knots))

    # get id of newly added line
    db_cursor.execute("SELECT LAST_INSERT_ROWID()")
    line_id = db_cursor.fetchall()[0][0]
   
    print("Placed line:")
    print_line_info(line_id, colorRGB, knots, len(detector_coords))

    # add detectors to database
    for n, (dx, dy) in enumerate(detector_coords):
        db_cursor.execute('''INSERT INTO detectors (number_on_line, x, y, w, h, line_id) 
                VALUES (?, ?, ?, ?, ?, ?)''',
                (n, dx, dy, detector_size[0], detector_size[1], line_id) )
    
    # save median_image
    db_cursor.execute("UPDATE median_image SET image = ? WHERE id = 2", (median_frame,) )

    # save and close database
    close_database(db_connection, db_cursor)
    
    # display median frame with line
    while True:
        cv.imshow("Median frame", median_frame)
        if cv.waitKey(0) & 0xFF == 27: break # until ESC is pressed
    
    # close windows
    cv.destroyAllWindows()

############################################################################################

def remove_line( namespace ):
    # lab_2.py remove_line <n>
    # -> removes line and all its detectors
    # then shows result
    db_connection, db_cursor = open_database()

    line_id = namespace.line_number
    
    # first check if request even exists
    db_cursor.execute("SELECT id FROM lines")
    rows = db_cursor.fetchall()
    line_ids = [row[0] for row in rows]

    if line_id not in line_ids:
        print("Error: line with id = {} does not exist".format(line_id))
        os._exit(os.EX_OK)
        
    # delete line
    print("Deleted line with id = {}".format(line_id))
    db_cursor.execute("DELETE FROM lines WHERE id = ?", (line_id,) )
    # due to relation this should also delete associated detectors

    # fetch median image from database
    db_cursor.execute("SELECT image FROM median_image WHERE id = 1")
    median_frame = db_cursor.fetchone()[0] 
    
    # redraw all remaining lines and detectors
    db_cursor.execute("SELECT * from lines")
    lines = db_cursor.fetchall()
    for line in lines:
        line_id = line[0]
        knots = line[4]
        colorBGR = (line[3], line[2], line[1])
        # redraw line
        draw_line(median_frame, knots, colorBGR)
        # redraw detectors
        db_cursor.execute("SELECT number_on_line, x, y, w, h FROM detectors WHERE line_id = ?", (line_id,))
        detectors = db_cursor.fetchall()
        for detector in detectors:
            draw_detector(median_frame, detector[0], 
                    (detector[1], detector[2]), (detector[3], detector[4]), colorBGR)
        
    print("Updating database")
    # save median_image
    db_cursor.execute("UPDATE median_image SET image = ? WHERE id = 2", (median_frame,) )

    # save and close database
    close_database(db_connection, db_cursor)

    # make window
    cv.namedWindow("Median frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("Median frame", 800, 600) # set window size
    # display median frame
    while True:
        cv.imshow("Median frame", median_frame)
        if cv.waitKey(0) & 0xFF == 27: break # until ESC is pressed
    # close windows
    cv.destroyAllWindows()

############################################################################################

def compute_average_for_detector(frame_gs, detector_coords, detector_size):
    dx_l = max( 0, int(detector_coords[0] - detector_size[0]/2) )
    dx_u = min( frame_gs.shape[1], int(detector_coords[0] + detector_size[0]/2) )
    dy_l = max( 0, int(detector_coords[1] - detector_size[1]/2) )
    dy_u = min( frame_gs.shape[0], int(detector_coords[1] + detector_size[1]/2) )
    detector_zone = frame_gs[ dy_l:dy_u, dx_l:dx_u ]
    return np.mean(detector_zone, axis=(0, 1))

from scipy import signal
from filterpy.gh import GHFilter

def filter_binarized(data):
    f = GHFilter (x=data[0], dx=0., dt=1., g=.4, h=.2)
    filtered = f.batch_filter(data)
    filtered[ filtered > 0.5 ]  = 1
    filtered[ filtered <= 0.5 ] = 0
    return filtered[1:, 0]

def compute_detections(binarized):
    detections = np.zeros(binarized.shape)
    pos = np.where( np.diff(binarized) > 0 )[0]
    detections[pos] = 1
    return detections

def run_detection( namespace ):
    #

    db_connection, db_cursor = open_database()

    # fetch median image from database
    db_cursor.execute("SELECT image FROM median_image WHERE id = 1")
    median_frame = db_cursor.fetchone()[0] 
    
    # fetch detectors
    db_cursor.execute("SELECT line_id, number_on_line, x, y, w, h FROM detectors")
    detectors = db_cursor.fetchall()
    
    detectors_data = { detector : ( # a dictionary of detector data
        [], # median frame average color
        [], # mean color
        [], # binarized mean color
        ) for detector in detectors}
    
    # fill median frame average color
    median_frame_gs = cv.cvtColor(median_frame, cv.COLOR_BGR2GRAY)
    for detector in detectors:
        detectors_data[detector][0].append(
                compute_average_for_detector(median_frame_gs, 
                    (detector[2], detector[3]), 
                    (detector[4], detector[5]) )
                )

    # fetch file list
    db_cursor.execute("SELECT * FROM file_list")
    file_list = [row[0] for row in db_cursor.fetchall()]
    
    # run detection
    for file in file_list:
        print("Running detection for {}".format(file))
        with video_capture_wrapper(file) as vid:
            while True:
                ret, frame = vid.read()
                if not ret: break
                frame_gs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # get grayscale of frame
                for detector in detectors:
                    median_frame_color = detectors_data[detector][0][0]
                    average_frame_color = compute_average_for_detector(frame_gs,
                            (detector[2], detector[3]), (detector[4], detector[5]) )
                    detectors_data[detector][1].append(average_frame_color)
                    if abs( average_frame_color - median_frame_color ) >= 3:
                        detectors_data[detector][2].append(1) # car detected
                    else:
                        detectors_data[detector][2].append(0) # no car
   
    # process and save data to database
    print("Processing data and saving to database")
    for detector in detectors:
        median_frame_color = detectors_data[detector][0][0]
        mean_color = np.array(detectors_data[detector][1])
        binarized = filter_binarized( np.array(detectors_data[detector][2]) )
        detections = compute_detections( binarized )
        db_cursor.execute('''
                UPDATE detectors 
                SET 
                    median_frame_color = ?,
                    mean_color = ?,
                    binarized = ?,
                    detections = ?
                WHERE 
                    line_id = ? AND number_on_line = ?
                ''',
                (median_frame_color, mean_color, binarized, detections, detector[0], detector[1]) )
    
    # save and close database
    close_database(db_connection, db_cursor)

############################################################################################

import matplotlib.pyplot as plt

def plot_detector( namespace ):
    #

    line_id = namespace.line_number
    detector_number = namespace.detector_number
    
    db_connection, db_cursor = open_database()
    
    # fetch detector data
    db_cursor.execute('''SELECT median_frame_color, mean_color, binarized, detections 
            FROM detectors WHERE line_id = ? AND number_on_line = ?''',
            (line_id, detector_number))
    detector = db_cursor.fetchone()

    # fetch fps
    db_cursor.execute("SELECT fps FROM mean_fps")
    fps = db_cursor.fetchone()[0]

    # save and close database
    close_database(db_connection, db_cursor)

    if detector is None:
        print("Error retrieving data for detector with line_id = {} and number_on_line = {}.".format(
            line_id, detector_number) )
        print("Does it exist?")
        os._exit(os.EX_OK)
    
    median_frame_color = detector[0]
    mean_color = detector[1]
    binarized = detector[2]
    detections = detector[3]
   
    t = np.arange(0, binarized.shape[0]) / fps

    # plot data
    ax1 = plt.subplot(3, 1, 1)
    ax1.title.set_text("mean color")
    ax1.plot(t, mean_color, 'b')
    ax1.axhline(y = median_frame_color, color='b', linestyle='--')
    ax1.set_xlabel('t, [s]', loc='right')
    ax1.grid()

    ax2 = plt.subplot(3, 1, 2)
    ax2.title.set_text("binarized")
    ax2.step(t, binarized, 'b', where='mid')
    ax2.set_xlabel('t, [s]', loc='right')
    ax2.grid()

    ax3 = plt.subplot(3, 1, 3)
    ax3.title.set_text("detections")
    ax3.step(t, np.cumsum(detections), 'b', where='mid')
    ax3.set_xlabel('t, [s]', loc='right')
    ax3.grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()

############################################################################################

import itertools
from scipy import signal
from filterpy.gh import GHFilter

def gh_filter(data):
    f = GHFilter (x=data[0], dx=0., dt=1., g=.05, h=.0001)
    filtered = f.batch_filter(data)
    return filtered[1:, 0]

def plot_line( namespace ):
    line_id = namespace.line_number
    
    db_connection, db_cursor = open_database()
    
    # fetch detectors data
    db_cursor.execute("SELECT * FROM detectors WHERE line_id = ?", (line_id,) )
    detectors = db_cursor.fetchall()

    # fetch fps
    db_cursor.execute("SELECT fps FROM mean_fps")
    fps = db_cursor.fetchone()[0]

    # fetch transform_matrix
    db_cursor.execute("SELECT matrix FROM transform_matrix WHERE id = -1")
    T_inv = db_cursor.fetchone()[0]

    # save and close database
    close_database(db_connection, db_cursor)

    if detectors is None: # exit if got nothing
        print("Error retrieving data for line with id = {}".format(line_id) )
        print("Does it exits?")
        os._exit(os.EX_OK)
    if detectors[0]['detections'] is None:
        print("Error retrieving detection data")
        print("Run run_detection first")
        os._exit(os.EX_OK)

    # compute velocity
    slice_window = int(fps) * 5
    v = np.zeros( detectors[0]['mean_color'].shape )

    for i, _ in enumerate(v):
        vs = []
        for d1, d2 in itertools.pairwise(detectors):
            #if d1['binarized'][i] == 1 and d2['binarized'][i] == 1:
            if True:
                mean_color_slice1 = d1['mean_color'] \
                    [ max(0, i-slice_window) : min(v.shape[0], i+slice_window)] - d1['median_frame_color']
                mean_color_slice2 = d2['mean_color'] \
                    [ max(0, i-slice_window) : min(v.shape[0], i+slice_window)] - d2['median_frame_color']
                correlation = signal.correlate(mean_color_slice1, mean_color_slice2, mode="full")
                lags = signal.correlation_lags(mean_color_slice1.size, mean_color_slice2.size, mode="full")
                pos = np.argmax(correlation)
                if pos == 0:
                    lag = 0
                else:
                    lag = -lags[np.argmax(correlation)]
                uv = np.array([
                    [d1['x'], d1['y']],
                    [d2['x'], d2['y']], 
                    ])
                xy = transform(uv, T_inv)
                dr = np.array([xy[0,0] - xy[1,0], xy[0,1] - xy[1,1]])
                dr = np.sqrt( dr.dot(dr) )
                if lag != 0:
                    vs.append( np.abs(dr / lag * fps))
        if vs:
            v[i] = np.mean(vs)
    
    # compute line length
    line_length = []
    for d1, d2 in itertools.pairwise(detectors):
        uv = np.array([
            [d1['x'], d1['y']],
            [d2['x'], d2['y']], 
            ])
        xy = transform(uv, T_inv)
        dr = np.array([xy[0,0] - xy[1,0], xy[0,1] - xy[1,1]])
        dr = np.sqrt( dr.dot(dr) )
        line_length.append(dr)
    line_length = np.sum(line_length)

    # compute intensity
    intensity = np.zeros( detectors[0]['mean_color'].shape )
    for i, _ in enumerate(intensity):
        if i == 0:
            intensity[0] = 0
            continue
        detections = [
                np.sum( detector['detections'][ max(0, i-slice_window): i ] ) / min(slice_window, i)
                for detector in detectors]
        intensity[i] = np.mean(detections) * fps
    
    # compute density
    rho = np.zeros( detectors[0]['mean_color'].shape )
    for detector in detectors:
        rho = rho + detector['binarized']
    rho = rho / line_length
    
    # plot data
   
    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits( (-1,1) ) 

    t = np.arange(0, v.shape[0]) / fps

    ax1 = plt.subplot(3, 1, 1)
    ax1.title.set_text("velocity")
    ax1.plot(t, v, 'b--', label="velocity")
    ax1.plot(t, gh_filter(v), 'b', label="filtered velocity")
    ax1.set_xlabel('t, [s]', loc='right')
    plt.ylabel("y", rotation=0)
    ax1.set_ylabel('v, [m/s]', loc='top')
    #ax1.set_ylim([0, 100])
    ax1.grid()

    ax2 = plt.subplot(3, 1, 2)
    ax2.title.set_text("intensity")
    ax2.plot(t,  gh_filter(v) * gh_filter(rho), 'b', label=r'$\rho_f \cdot v_f$')
    ax2.plot(t, intensity, 'g--', label='intensity')
    ax2.plot(t, gh_filter(intensity), 'g', label="filtered intensity")
    ax2.set_xlabel('t, [s]', loc='right')
    plt.ylabel("y", rotation=0)
    ax2.set_ylabel('q, [1/s]', loc='top')
    ax2.yaxis.set_major_formatter(formatter)
    ax2.grid()
    #ax2.legend()
    
    ax3 = plt.subplot(3, 1, 3)
    ax3.title.set_text("density")
    ax3.plot(t, rho, 'b--')
    ax3.plot(t, gh_filter(rho), 'b', label="filtered density")
    ax3.set_xlabel('t, [s]', loc='right')
    plt.ylabel("y", rotation=0)
    ax3.set_ylabel('ρ, [1/m]', loc='top')
    ax3.yaxis.set_major_formatter(formatter)
    ax3.grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()

############################################################################################

def set_marking( namespace ):

    if not hasattr(namespace, 'method'):
        print("To set marking a method must be specified.")
        os._exit(os.EX_OK)
    if namespace.method == "rectangle":
        xy = [
                [0, 0],
                [namespace.a, 0],
                [namespace.a, namespace.b],
                [0, namespace.b]
                ]
    
    if namespace.method == "rectangle_mesh":
        print("Not implemented(((((")
        os._exit(os.EX_OK)

    if namespace.method == "coordinates":
        r = namespace.r
        xy = []
        if len(r) % 2 == 1:
            print("Incorrect number of coordinates")
            os._exit(os.EX_OK)
        while len(r) > 0:
            x = r.pop(0)
            y = r.pop(0)
            xy.append([x, y])

    db_connection, db_cursor = open_database()
    
    # fetch median image from database
    db_cursor.execute("SELECT image FROM median_image WHERE id = 1")
    median_frame = db_cursor.fetchone()[0] 

    # make window
    cv.namedWindow("Median frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("Median frame", 800, 600) # set window size

    uv = []
    # mouse callback function
    def add_point(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if len(uv) < len(xy):
                uv.append( (x, y) ) # add to list
                draw_knot(median_frame, x, y, (0,255,0))
                cv.imshow("Median frame", median_frame)

    # add mouse callback to window
    cv.setMouseCallback("Median frame", add_point)

    # display median frame
    while True:
        cv.imshow("Median frame", median_frame)
        if cv.waitKey(0) & 0xFF == 27: break # until ESC is pressed

    print("Click on image to place a point")
    print("Press ESC to finish")

    T, err = dlt(np.array(xy), np.array(uv), normalization=True)
    print("DLT error: {}".format(err))
    T_inv = np.linalg.inv(T)

    # fetch median image from database
    db_cursor.execute("SELECT image FROM median_image WHERE id = 1")
    median_frame = db_cursor.fetchone()[0] 

    # update transform matrix in database
    db_cursor.executemany("UPDATE transform_matrix SET matrix = ? WHERE id = ?",
            [(T, 1), (T_inv, -1)] )
    print("Transform matrix:"); print(T)
    
    median_frame = draw_markings(median_frame, T, T_inv)

    # save median_image
    db_cursor.execute("UPDATE median_image SET image = ? WHERE id = 3", (median_frame,) )
    
    # save and close database
    close_database(db_connection, db_cursor)

    # unset mouse callback
    cv.setMouseCallback("Median frame", lambda event, x, y, flags, parameters : None)

    # display median frame
    while True:
        cv.imshow("Median frame", median_frame)
        if cv.waitKey(0) & 0xFF == 27: break # until ESC is pressed
    # close windows
    cv.destroyAllWindows()

############################################################################################

def show_marking( namespace ):
    db_connection, db_cursor = open_database()

    # fetch median image from database
    db_cursor.execute("SELECT image FROM median_image WHERE id = 3")
    median_frame = db_cursor.fetchone()[0] 

    # fetch transform matrix form database
    db_cursor.execute("SELECT matrix FROM transform_matrix WHERE id = 1")
    T = db_cursor.fetchone()[0]
    print("Transform matrix:"); print(T)

    # save and close database
    close_database(db_connection, db_cursor)

    # make window
    cv.namedWindow("Median frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("Median frame", 800, 600) # set window size
    # display median frame
    while True:
        cv.imshow("Median frame", median_frame)
        if cv.waitKey(0) & 0xFF == 27: break # until ESC is pressed
    # close windows
    cv.destroyAllWindows()

############################################################################################

def reset_marking( namespace ):
    db_connection, db_cursor = open_database()

    # fetch median image from database
    db_cursor.execute("SELECT image FROM median_image WHERE id = 1")
    median_frame = db_cursor.fetchone()[0] 

    # set transform matrix to default
    T = np.diag([median_frame.shape[1], median_frame.shape[0], 1])
    T_inv = np.diag([1/median_frame.shape[1], 1/median_frame.shape[0], 1])

    # update transform matrix in database
    db_cursor.executemany("UPDATE transform_matrix SET matrix = ? WHERE id = ?",
            [(T, 1), (T_inv, 1)] )
    print("Transform matrix:"); print(T)
    
    median_frame = draw_markings(median_frame, T, T_inv)

    # save median_image
    db_cursor.execute("UPDATE median_image SET image = ? WHERE id = 3", (median_frame,) )

    # save and close database
    close_database(db_connection, db_cursor)

    # make window
    cv.namedWindow("Median frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("Median frame", 800, 600) # set window size
    # display median frame
    while True:
        cv.imshow("Median frame", median_frame)
        if cv.waitKey(0) & 0xFF == 27: break # until ESC is pressed
    # close windows
    cv.destroyAllWindows()

############################################################################################

if __name__ == "__main__":
    from numpy.random import default_rng
    rng = default_rng()
    x = rng.standard_normal(1000)
    y = np.concatenate([rng.standard_normal(100), x]) * 0
    correlation = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    pos = np.argmax(correlation)
    if pos == 0:
        lag = 0
    else:
        lag = lags[np.argmax(correlation)]
    print(lag)
    pass
