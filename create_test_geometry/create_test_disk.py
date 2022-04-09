#!/usr/bin/env python3

import numpy as np
from math import cos, sin
from math import pi as PI
from wavefront_obj_writer import write_wavefront_obj

TEST_DISK_BASENAME = "test_disk"

RADIUS = 1.0
TRACKS = 11
MINOR_TICKS = 11

# Wavefront obj format
START_INDEX = 1

def get_lists_length(dic):
    values = list(dic.values())
    length = len(values[0])
    for value in values:
        assert(length == len(value))
    return length

def append(vertices, **kwargs):
    for key, value in kwargs.items():
        vertices[key].append(value)

def list_track_vertices(vertices, track):
    points = vertices["points"]
    distances = vertices["distances"]
    tracks = vertices["tracks"]
    indices = vertices["indices"]
    length = get_lists_length(vertices)
    track_vertices = {"points": [], "distances": [], "indices": []}
    for i in range(length):
        if (tracks[i] == track):
            append(track_vertices, points=points[i], distances=distances[i], indices=indices[i])
    return track_vertices

def z_rotation_matrix(radians):
    return np.array([
        [ cos(radians), -sin(radians),  0],
        [ sin(radians),  cos(radians),  0],
        [ 0,             0,             1]
    ])

def add_vertices_quarter(vertices, point, distance, track, index):
    tick_rotation_radians = -(1/(2 * (MINOR_TICKS + 1))) * PI
    for minor_tick in range(1, MINOR_TICKS + 1):
        index += 1
        radians = minor_tick * tick_rotation_radians
        rotation_matrix = z_rotation_matrix(radians)
        minor_point = np.dot(rotation_matrix, point)
        append(vertices, points=minor_point, distances=distance, tracks=track, indices=index)
    return index

def add_vertices(vertices, radius, track, index):
    distance = radius
    # West vertex
    index += 1
    point = np.array([-radius, 0, 0])
    append(vertices, points=point, distances=distance, tracks=track, indices=index)
    # North-West quarter
    index = add_vertices_quarter(vertices=vertices, point=point, distance=distance, track=track, index=index)
    # North vertex
    index += 1
    point = np.array([0, radius, 0])
    append(vertices, points=point, distances=distance, tracks=track, indices=index)
    # North-East quarter
    index = add_vertices_quarter(vertices=vertices, point=point, distance=distance, track=track, index=index)
    # East vertex
    index += 1
    point = np.array([radius, 0, 0])
    append(vertices, points=point, distances=distance, tracks=track, indices=index)
    # South-East quarter
    index = add_vertices_quarter(vertices=vertices, point=point, distance=distance, track=track, index=index)
    # South vertex
    index += 1
    point = np.array([0, -radius, 0])
    append(vertices, points=point, distances=distance, tracks=track, indices=index)
    # South-West quarter
    index = add_vertices_quarter(vertices=vertices, point=point, distance=distance, track=track, index=index)
    return index

def create_vertices():
    track_distance = RADIUS / TRACKS
    vertices = {"points": [], "distances": [], "tracks": [], "indices": []}
    index = START_INDEX
    append(vertices, points=np.zeros(3), distances=0.0, tracks=0, indices=index)
    for track in range(1, TRACKS):
        index = add_vertices(vertices=vertices, radius=track * track_distance, track=track, index=index)
    index = add_vertices(vertices=vertices, radius=RADIUS, track=TRACKS, index=index)
    assert(get_lists_length(vertices) == (index + 1 - START_INDEX))
    return vertices

def create_faces(vertices):
    faces = []
    track_0_vertex = list_track_vertices(vertices=vertices, track=0)
    assert(get_lists_length(track_0_vertex) == 1)
    track_0_index, = track_0_vertex["indices"]
    track_1_vertices = list_track_vertices(vertices=vertices, track=1)
    track_1_indices = track_1_vertices["indices"]
    track_1_length = get_lists_length(track_1_vertices)
    for i in range(track_1_length):
        face = [track_0_index, track_1_indices[i], track_1_indices[i - 1]]
        faces.append(face)
    prev_track_vertices = None
    prev_track_indices = None
    prev_track_length = None
    track_vertices = track_1_vertices
    track_indices = track_1_indices
    track_length = track_1_length
    for track in range(2, TRACKS + 1):
        prev_track_vertices = track_vertices
        prev_track_indices = track_indices
        prev_track_length = track_length
        track_vertices = list_track_vertices(vertices=vertices, track=track)
        track_indices = track_vertices["indices"]
        track_length = get_lists_length(track_vertices)
        assert(track_length == prev_track_length)
        for i in range(track_length):
            face1 = [prev_track_indices[i], track_indices[i], prev_track_indices[i - 1]]
            face2 = [prev_track_indices[i - 1], track_indices[i], track_indices[i - 1]]
            faces.append(face1)
            faces.append(face2)
    return faces

def create_disk():
    vertices = create_vertices()
    faces = create_faces(vertices=vertices)
    return vertices, faces

def main():
    vertices, faces = create_disk()
    write_wavefront_obj(basename=TEST_DISK_BASENAME, points=vertices["points"], distances=vertices["distances"], faces=faces)

if __name__ == "__main__":
    main()