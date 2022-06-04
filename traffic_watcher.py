#!/usr/bin/env python

if __name__ == "__main__":
    
    import argparse
    my_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS) # create ArgumentParser object

    my_subparsers = my_parser.add_subparsers() # create subparsers 

    ###############################################################################################
    # create subparsers for evety command

    # lab_2.py new_session <list of .mp4's>
    # -> creates new session for detection (overwrites old)
    # gets list of files, checks for availability,
    # writes them into DB and exits;
    # also creates median_frame for later use and store to DB
    
    new_session_parser = my_subparsers.add_parser('new_session', help =
            '''Starts a new detection session. Checks all files and resets app's database. Also approximates background (calculates median frame) and displays it.'''
            )
    new_session_parser.add_argument('files', type=str, nargs='+', metavar='<.mp4 files>', help = 
            '''List of .mp4 files with traffic''')
   
    # set func to new_session() for this subparser
    from commands import new_session
    new_session_parser.set_defaults( func=new_session )

    # lab_2.py show_detectors
    # -> shows median_frame with lines and detectors
    # shows line and detector numbers
    # exits on ESC

    show_detectors_parser = my_subparsers.add_parser('show_detectors', help = 
            '''Show current lines and detectors. Also prints info in console.''')
    # no args to add

    # set func to show_detectors() for this subparser
    from commands import show_detectors
    show_detectors_parser.set_defaults( func=show_detectors )

    # lab_2.py place_line --colorRGB <R> <G> <B> --detector_size <W> <H>
    # -> shows median_frame with lines and detectors
    # allow to place knots on image
    # after ESC interpolates line and places detectors
    # exits on second ESC

    place_line_parser = my_subparsers.add_parser('place_line', help = 
            '''Start a line placing procedure.''')
    place_line_parser.add_argument('--colorRGB', type=int, 
            nargs=3, metavar='C', default=None, help = '''Color of line. By default picks color automatically.''')
    place_line_parser.add_argument('--detector_size', type=int, 
            nargs=2, metavar='S', default=[40, 30], help = '''Detectors' width and height. Default values are width = 40 and height = 30''')
    
    # set func to place_line for this subparser
    from commands import place_line
    place_line_parser.set_defaults( func=place_line )

    # lab_2.py remove_line <n>
    # -> removes line and all its detectors
    # then shows result

    remove_line_parser = my_subparsers.add_parser('remove_line', help = 
            '''Remove selected line.''')
    remove_line_parser.add_argument('line_number', type=int,  metavar='N', help = "Number of line to remove.")

    # set func to remove_line for this subparser
    from commands import remove_line
    remove_line_parser.set_defaults( func=remove_line )

    # lab_2.py run_detection
    #

    run_detection_parser = my_subparsers.add_parser('run_detection', help = 
            '''Run detection for session. Saves all detection data to database making it available to "plot_detector" and "plot_line" commands.''')
    # no args to add

    # set func to run_detection for this subparser
    from commands import run_detection
    run_detection_parser.set_defaults( func=run_detection )

    # lab_2.py plot_detector <line_number=1> <detector_number=0>
    # 

    plot_detector_parser = my_subparsers.add_parser('plot_detector', help =
            '''Plots detection data for selected detector. ''')
    plot_detector_parser.add_argument('line_number', type=int, metavar='L', default=1, nargs='?', help = "Line number")
    plot_detector_parser.add_argument('detector_number', type=int, metavar='D', default=0, nargs='?', help = "Detector number")

    # set func to plot_detector for this subparser
    from commands import plot_detector
    plot_detector_parser.set_defaults( func=plot_detector )

    # lab_2.py plot_line <line_number=1>
    # 

    plot_line_parser = my_subparsers.add_parser('plot_line', help = 
            '''Plots detection data for selected line. Approximates velocity, density and intensity.''')
    plot_line_parser.add_argument('line_number', type=int, metavar='L', default=1, nargs='?', help = "Line number")

    # set func to plot_line for this subparser
    from commands import plot_line
    plot_line_parser.set_defaults( func=plot_line )

    # lab_2.py set_marking
    set_marking_parser = my_subparsers.add_parser('set_marking', help = "Set marking. Multiple ways are available.")
    
    set_marking_subparsers = set_marking_parser.add_subparsers() # create subparsers for this subparser

    # set func to set_marking for these subparsers
    from commands import set_marking
    set_marking_parser.set_defaults( func=set_marking )

    # lab_2.py set_marking rectangle a b
    set_marking_rectangle_subparser = set_marking_subparsers.add_parser('rectangle', help = "Set marking by edges of rectangle.")
    set_marking_rectangle_subparser.add_argument('a', type=float, metavar='a', default=1.0, nargs='?', help = "Rectangle width.")
    set_marking_rectangle_subparser.add_argument('b', type=float, metavar='b', default=1.0, nargs='?', help = "Rectangle height.")
   
    # set method to "rectangle" in namespace
    set_marking_rectangle_subparser.set_defaults( method="rectangle" )

    # lab_2.py set_marking square_mesh
    set_marking_rectangle_mesh_subparser = set_marking_subparsers.add_parser('rectangle_mesh', 
            help = "Set marking by edges of rectangle mesh.")
    set_marking_rectangle_mesh_subparser.add_argument('a', type=float, metavar='a', default=1.0, nargs='?', 
            help = "Width of rectangle grid")
    set_marking_rectangle_mesh_subparser.add_argument('b', type=float, metavar='b', default=1.0, nargs='?', 
            help = "Height of rectangle grid")
   
    # set method to "rectangle_mesh" in namespace
    set_marking_rectangle_mesh_subparser.set_defaults( method="rectangle_mesh" )

    # lab_2.py set_marking coordinates
    set_marking_coordinates_subparser = set_marking_subparsers.add_parser('coordinates', 
            help = "Set marking by points with known coordinates.")
    set_marking_coordinates_subparser.add_argument('r', type=float, metavar='xi, yi', nargs='+', 
            help = "Coordinates of points")
   
    # set method to "coordinates" in namespace
    set_marking_coordinates_subparser.set_defaults( method="coordinates" )

    # lab_2.py show_marking
    show_marking_parser = my_subparsers.add_parser('show_marking', help = 
            '''Show current marking lines. Also prints transform matrix in console.''')
    # no args to add

    # set func to show_marking() for this subparser
    from commands import show_marking
    show_marking_parser.set_defaults( func=show_marking )

    # lab_2.py reset_marking
    reset_marking_parser = my_subparsers.add_parser('reset_marking', help = 
            '''Reset marking to default.''')
    # no args to add

    # set func to reset_marking() for this subparser
    from commands import reset_marking
    reset_marking_parser.set_defaults( func=reset_marking )

    ###############################################################################################

    # parse sys.argv
    my_namespace = my_parser.parse_args()

    # execute func
    if hasattr(my_namespace, 'func'):
        my_namespace.func(my_namespace)
    else: # was called lab_2.py <nothing>
        # just show image then
        show_detectors(my_namespace)
