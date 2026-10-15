from typing import Union, Tuple, List
from .buffers import ReadableBuffer
from enum import Enum
import os
import trimesh

class ShaderType(Enum):
    OPAQUE = 0
    TRANSPARENT = 1
    EMISSIVE = 2
    LAVA = 3

class Vec3:

    x: float
    y: float
    z: float

    def __init__(self, *args: Union[None, Tuple[float, float, float]]):

        if len(args) == 0:
            self.x = 0
            self.y = 0
            self.z = 0
            return

        if len(args) == 1:
            # Tuple specified
            self.x, self.y, self.z = args[0]
            return

        if len(args) == 3:
            #Individual arguments specified
            self.x, self.y, self.z = args
            return

        raise ValueError('Unknown method to instantiate the Vector')

    @staticmethod
    def from_buffer(buffer: ReadableBuffer):

        return Vec3(
            buffer.parse_float(),
            buffer.parse_float(),
            buffer.parse_float(),
        )

    @property
    def tup(self):
        return (self.x, self.y, self.z)

class Tri:

    a: int
    b: int
    c: int

    def __init__(self, *args: Union[None, Tuple[int, int, int]]):

        if len(args) == 1:
            # Tuple specified
            self.a, self.b, self.c = args[0]

        elif len(args) == 3:
            # Individual arguments specified
            self.a, self.b, self.c = args

        else:
            raise ValueError('Unknown method to instantiate Tri')

        if self.a < 0:
            raise ValueError("Tri A index out of range.")

        if self.b < 0:
            raise ValueError("Tri B index out of range.")

        if self.c < 0:
            raise ValueError("Tri C index out of range.")

    @staticmethod
    def from_buffer(buffer: ReadableBuffer):

        return Tri(
            buffer.parse_ushort(),
            buffer.parse_ushort(),
            buffer.parse_ushort(),
        )

    @property
    def tup(self):
        return (self.a, self.b, self.c)

class RGBA:

    r: int
    g: int
    b: int
    a: int

    def __init__(self, *args):

        if len(args) == 0:
            self.r = 0
            self.g = 0
            self.b = 0
            self.a = 0

        elif len(args) == 1:
            # Tuple specified
            self.r, self.g, self.b, self.a = args[0]

        elif len(args) == 4:
            # Individual arguments specified
            self.r, self.g, self.b, self.a = args

        else:
            raise ValueError('Unknown method to instantiate RGBA')

        if self.r < 0  or self.r > 255:
            raise ValueError("RGBA R value out of range.")

        if self.g < 0  or self.g > 255:
            raise ValueError("RGBA G value out of range.")

        if self.b < 0  or self.b > 255:
            raise ValueError("RGBA B value out of range.")

        if self.a < 0  or self.a > 255:
            raise ValueError("RGBA A value out of range.")

    @staticmethod
    def from_buffer(buffer: ReadableBuffer):

        return RGBA(
            buffer.parse_byte(),
            buffer.parse_byte(),
            buffer.parse_byte(),
            buffer.parse_byte(),
        )

    @property
    def tup(self):
        return (self.r, self.g, self.b, self.a)

class Vertex:

    position: Vec3
    color: RGBA
    normal: Vec3

    def __init__(self, position: Vec3 = Vec3(), color: RGBA = RGBA(), normal: Vec3 = Vec3()):

        self.position = Vec3(-position.x, position.y, position.z)
        self.color = color
        self.normal = normal

    @staticmethod
    def from_buffer(buffer: ReadableBuffer):

        return Vertex(
            Vec3.from_buffer(buffer),
            RGBA.from_buffer(buffer),
            Vec3.from_buffer(buffer),
        )

class Mesh:
    '''
    Contains raw triangle / vertex / normal data.
    '''

    verts: List[Vertex]
    tris: List[Tri]

    def __init__(self, verts: List[Vertex] = [], tris: List[Vertex] = []):

        self.verts = verts.copy()
        self.tris = tris.copy()

    @staticmethod
    def from_buffer(buffer: ReadableBuffer):

        num_verts = buffer.parse_ushort()
        buffer.get(4) # No clue what these bytes do.

        assert num_verts > 0

        verts = [Vertex.from_buffer(buffer) for _ in range(num_verts)]

        num_tris = buffer.parse_signed_int()

        assert num_tris > 0

        tris = [Tri.from_buffer(buffer) for _ in range(num_tris // 3)]

        return Mesh(verts, tris)


class Submesh:

    '''
    Contains a reference to the source mesh, and the start and end indices.
    '''

    source_mesh: Mesh

    buffer_start: int
    buffer_length: int

    shader: ShaderType

    bounds_min: Vec3
    bounds_max: Vec3

    name: str

    def __init__(self, source_mesh: Mesh, buffer_start: int, buffer_length: int, shader: ShaderType, bounds_min: Vec3, bounds_max: Vec3, name: str):

        self.source_mesh = source_mesh
        
        self.buffer_start = buffer_start
        self.buffer_length = buffer_length

        self.shader = shader

        self.bounds_min = bounds_min
        self.bounds_max = bounds_max

    @staticmethod
    def from_buffer(source_mesh: Mesh, buffer: ReadableBuffer):

        buffer_start = buffer.parse_uint()
        buffer_length = buffer.parse_uint()

        buffer.get(2) # Unknown

        shader = ShaderType(buffer.parse_ushort())

        bounds_min = Vec3.from_buffer(buffer)

        bounds_max = Vec3.from_buffer(buffer)

        buffer.get(2) # Unknown

        name_length = buffer.parse_ushort()

        name = buffer.parse_string(name_length)

        buffer.get(3 * 4) # Unknown vec

        return Submesh(
            source_mesh, 
            buffer_start, 
            buffer_length,
            shader,
            bounds_min,
            bounds_max, 
            name
        )

    def to_trimesh(self):

        # For this, we need to isolate the vertices of this specific submesh in as efficient of a manner as possible.
        # I think the best way to do this is to loop over the verts as I see them and build up a mapping to the new verts, 
        # and add those verts in.

        positions = []
        normals = []
        colors = []

        index_lookup = {}

        faces = []

        def apply_lookup(index):

            result = index_lookup.get(index, None)

            if result is not None:
                return result

            new_index = len(index_lookup)
            index_lookup[index] = new_index

            # Load the real data for this index.
            vert = self.source_mesh.verts[index]

            positions.append(vert.position.tup)
            normals.append(vert.normal.tup)
            colors.append(vert.color.tup)

            return new_index

        for index_location in range(self.buffer_start, self.buffer_start + self.buffer_length, 3):

            tri_location = index_location // 3

            tri = self.source_mesh.tris[tri_location]

            mapped_a = apply_lookup(tri.a)
            mapped_b = apply_lookup(tri.b)
            mapped_c = apply_lookup(tri.c)

            faces.append((mapped_a, mapped_b, mapped_c))

        return trimesh.Trimesh(
            vertices = positions,
            faces = faces,
            vertex_normals = normals,
            vertex_colors = colors,
            process = False
        )

class SWMesh:

    '''
    Submesh container.
    '''

    mesh: Mesh
    submeshes: List[Submesh]

    def __init__(self, mesh: Mesh = Mesh(), submeshes: List[Submesh] = []):

        self.mesh = mesh
        self.submeshes = submeshes.copy()

    @staticmethod
    def from_buffer(buffer: ReadableBuffer):

        buffer.get(8) # Skip the header.

        mesh = Mesh.from_buffer(buffer)

        num_submesh = buffer.parse_ushort()

        return SWMesh(
            mesh,
            [Submesh.from_buffer(mesh, buffer) for _ in range(num_submesh)]
        )

    @staticmethod
    def from_file(file_path: str):

        if not os.path.exists(file_path):
            raise FileNotFoundError('The specified mesh file does not exist:')

        with open(file_path, 'rb') as f:

            buffer = ReadableBuffer(f.read(), 'little')

        return SWMesh.from_buffer(buffer)

if __name__ == '__main__':

    m = SWMesh.from_file('island_12.mesh')

    m.submeshes[0].to_trimesh().export('test.ply')
