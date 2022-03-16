import pyglGA.ECSS.utilities as util
import numpy as np

class IndexedConverter():
    
    # Assumes triangulated buffers. Produces indexed results that support
    # normals as well.
    def Convert(self, vertices, colors, indices, produceNormals=True):

        iVertices = [];
        iColors = [];
        iNormals = [];
        iIndices = [];
        for i in range(0, len(indices), 3):
            iVertices.append(vertices[indices[i]]);
            iVertices.append(vertices[indices[i + 1]]);
            iVertices.append(vertices[indices[i + 2]]);
            iColors.append(colors[indices[i]]);
            iColors.append(colors[indices[i + 1]]);
            iColors.append(colors[indices[i + 2]]);
            iNormals.append(util.calculateNormals(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));
            iNormals.append(util.calculateNormals(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));
            iNormals.append(util.calculateNormals(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));

            iIndices.append(i);
            iIndices.append(i + 1);
            iIndices.append(i + 2);

        iVertices = np.array(
            iVertices,
            dtype=np.float32
        )
        iColors = np.array(
            iColors,
            dtype=np.float32
        )
        iNormals = np.array(
            iNormals,
            dtype=np.float32
        )
        iIndices = np.array(
            iIndices,
            dtype=np.uint32
        );

        return iVertices, iColors, iNormals, iIndices;