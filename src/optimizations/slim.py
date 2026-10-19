import trimesh
import numpy as np

def remove_small_faces(mesh: trimesh.Trimesh, threshold = 1) -> trimesh.Trimesh:

    face_areas = mesh.area_faces

    keep = face_areas > threshold

    mesh.update_faces(keep)

    mesh.remove_unreferenced_vertices()

    return mesh

def remove_buried_triangles(mesh: trimesh.Trimesh) -> trimesh.Trimesh:

    # Get the triangle centroids
    centroids = mesh.triangles_center
    count = len(mesh.faces)

    origins = centroids.copy()
    origins[:, 1] = mesh.bounds[1, 1] + 1000.0

    directions = np.zeros((count, 3))
    directions[:, 1] = -1

    hits = mesh.ray.intersects_first(ray_origins = origins, ray_directions = directions)

    keep = hits == np.arange(count)

    print(f'Original triangles: {count}')
    print(f'Kept Triangles: {np.sum(keep)}')
    print(f'Removed Triangles: {np.sum(~keep)}')

    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()

    return mesh