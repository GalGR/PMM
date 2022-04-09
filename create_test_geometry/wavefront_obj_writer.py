#!/usr/bin/env python3

def write_wavefront_obj(basename, points, distances, faces):
    filename_obj = basename + ".obj"
    filename_txt = basename + ".txt"
    with open(filename_obj, "w") as file:
        file.write("# List of vertices\n")
        for point in points:
            file.write(f"v {point[0]} {point[1]} {point[2]}\n")
        file.write("# List of faces\n")
        for face in faces:
            file.write(f"f {face[0]} {face[1]} {face[2]}\n")
    # with open(filename, "")