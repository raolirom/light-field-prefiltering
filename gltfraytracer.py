import os
import numpy as np
import embree
from pygltflib import GLTF2
from PIL import Image

def dot(a, b):
    return np.einsum('...i,...i', a, b)

def srgb2lin(srgb):
        return np.where(srgb >= 0.04045, np.power((srgb + 0.055) / 1.055, 2.4), srgb/12.92)
    
def lin2srgb(lin):
    return np.where(lin > 0.0031308, 1.055 * np.power(lin, 1.0/2.4) - 0.055, 12.92 * lin)

class Material3D:
    def __init__(self, path, gltf, material):
        # Default values.
        self.base_color = np.ones((1, 1, 4))
        self.metallic = np.ones((1, 1, 1))
        self.roughness = np.ones((1, 1, 1))
        self.normal = np.ones((1, 1, 3)) * [0.0, 0.0, 1.0]

        # Load base color texture.
        baseColorTexture = material.pbrMetallicRoughness.baseColorTexture
        if baseColorTexture is not None:
            texture = gltf.textures[baseColorTexture.index]
            source_image = gltf.images[texture.source]
            with Image.open(os.path.join(path, source_image.uri)) as img:
                self.base_color = np.swapaxes(np.array(img.convert('RGBA')), 0, 1) / 255.0

        # Load metallic roughness texture.
        metallicRoughnessTexture = material.pbrMetallicRoughness.metallicRoughnessTexture
        if metallicRoughnessTexture is not None:
            texture = gltf.textures[metallicRoughnessTexture.index]
            source_image = gltf.images[texture.source]
            with Image.open(os.path.join(path, source_image.uri)) as img:
                metallic_roughness = np.swapaxes(np.array(img.convert('RGBA')), 0, 1) / 255.0
                self.metallic = metallic_roughness[..., 2:3]
                self.roughness = metallic_roughness[..., 1:2]

        # Load normal map texture.
        normalTexture = material.normalTexture
        if normalTexture is not None:
            texture = gltf.textures[normalTexture.index]
            source_image = gltf.images[texture.source]
            with Image.open(os.path.join(path, source_image.uri)) as img:
                self.normal = (np.swapaxes(np.array(img.convert('RGB')), 0, 1) / 255.0) * 2.0 - 1.0

    def fetch_base_color(self, texcoord):
        # TexCoord wrapping.
        texcoord = texcoord % 1.0

        # Fetch base color texture.
        resolution = np.array(self.base_color.shape[:-1])
        indices = np.minimum(texcoord * resolution, resolution - 1).astype(int)
        base_color = self.base_color[indices[:, 0], indices[:, 1]]

        return base_color
        
    def fetch_metallic_roughness(self, texcoord):
        # TexCoord wrapping.
        texcoord = texcoord % 1.0
        
        # Fetch metallic roughness texture.
        resolution = np.array(self.metallic.shape[:-1])
        indices = np.minimum(texcoord * resolution, resolution - 1).astype(int)
        metallic = self.metallic[indices[:, 0], indices[:, 1]]
        roughness = self.roughness[indices[:, 0], indices[:, 1]]

        return [metallic, roughness]
    
    def fetch_normal(self, texcoord):
        # TexCoord wrapping.
        texcoord = texcoord % 1.0

        # Fetch normal map texture.
        resolution = np.array(self.normal.shape[:-1])
        indices = np.minimum(texcoord * resolution, resolution - 1).astype(int)
        normal = self.normal[indices[:, 0], indices[:, 1]]

        return normal

class Mesh3D:
    def __init__(self, gltf, primitive, rotation, translation):
        self.position = None
        self.texcoord = None
        self.normal = None
        self.tangent = None
        self.bitangent = None
        self.indices = None
        self.material = primitive.material

        # Load position attribute.
        if primitive.attributes.POSITION is not None:
            accessor = gltf.accessors[primitive.attributes.POSITION]
            bufferView = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[bufferView.buffer]
            data = gltf.get_data_from_buffer_uri(buffer.uri)
            self.position = np.frombuffer(data, dtype='float32', count=3*accessor.count, offset=bufferView.byteOffset+accessor.byteOffset).reshape(accessor.count, 3)
            self.position = (self.position @ rotation.T) + translation

        # Load texcoord attribute.
        if primitive.attributes.TEXCOORD_0 is not None:
            accessor = gltf.accessors[primitive.attributes.TEXCOORD_0]
            bufferView = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[bufferView.buffer]
            data = gltf.get_data_from_buffer_uri(buffer.uri)
            self.texcoord = np.frombuffer(data, dtype='float32', count=2*accessor.count, offset=bufferView.byteOffset+accessor.byteOffset).reshape(accessor.count, 2)

        # Load normal attribute.
        if primitive.attributes.NORMAL is not None:
            accessor = gltf.accessors[primitive.attributes.NORMAL]
            bufferView = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[bufferView.buffer]
            data = gltf.get_data_from_buffer_uri(buffer.uri)
            self.normal = np.frombuffer(data, dtype='float32', count=3*accessor.count, offset=bufferView.byteOffset+accessor.byteOffset).reshape(accessor.count, 3)
            self.normal = self.normal @ rotation.T

        # Load tangent attribute.
        if primitive.attributes.TANGENT is not None:
            accessor = gltf.accessors[primitive.attributes.TANGENT]
            bufferView = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[bufferView.buffer]
            data = gltf.get_data_from_buffer_uri(buffer.uri)
            tangent = np.frombuffer(data, dtype='float32', count=4*accessor.count, offset=bufferView.byteOffset+accessor.byteOffset).reshape(accessor.count, 4)
            self.tangent = tangent[..., 0:3] @ rotation.T
            if self.normal is not None:
                self.bitangent = np.cross(self.normal, self.tangent) * tangent[..., 3:4]

        # Load vertex indices.
        if primitive.indices is not None:
            accessor = gltf.accessors[primitive.indices]
            bufferView = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[bufferView.buffer]
            data = gltf.get_data_from_buffer_uri(buffer.uri)
            self.indices = np.frombuffer(data, dtype='uint16', count=accessor.count, offset=bufferView.byteOffset+accessor.byteOffset).reshape(accessor.count//3, 3).astype(np.uint32)

    def interpolate_texcoord(self, prim_id, uv):
        indices = self.indices[prim_id]

        if self.texcoord is not None:
            texcoord = self.texcoord[indices, ...]
            texcoord = (1.0 - uv[..., np.newaxis, 0] - uv[..., np.newaxis, 1]) * texcoord[:, 0, :] + uv[..., np.newaxis, 0] * texcoord[:, 1, :] + uv[..., np.newaxis, 1] * texcoord[:, 2, :]
        else:
            texcoord = np.array([0.0, 0.0])

        return texcoord

    def interpolate_tbn(self, prim_id, uv):
        indices = self.indices[prim_id]
        position = self.position[indices, ...]

        if self.tangent is not None:
            tangent = self.tangent[indices, ...]
            tangent = (1.0 - uv[..., np.newaxis, 0] - uv[..., np.newaxis, 1]) * tangent[:, 0, :] + uv[..., np.newaxis, 0] * tangent[:, 1, :] + uv[..., np.newaxis, 1] * tangent[:, 2, :]
        else:
            tangent = position[:, 1, :] - position[:, 0, :] # V0->V1

        if self.bitangent is not None:
            bitangent = self.bitangent[indices, ...]
            bitangent = (1.0 - uv[..., np.newaxis, 0] - uv[..., np.newaxis, 1]) * bitangent[:, 0, :] + uv[..., np.newaxis, 0] * bitangent[:, 1, :] + uv[..., np.newaxis, 1] * bitangent[:, 2, :]
        else:
            bitangent = position[:, 2, :] - position[:, 0, :] # V0->V2

        if self.normal is not None:
            normal = self.normal[indices, ...]
            normal = (1.0 - uv[..., np.newaxis, 0] - uv[..., np.newaxis, 1]) * normal[:, 0, :] + uv[..., np.newaxis, 0] * normal[:, 1, :] + uv[..., np.newaxis, 1] * normal[:, 2, :]
        else:
            normal = np.cross(tangent, bitangent)

        normal /= np.sqrt(dot(normal, normal))[:, np.newaxis] # normalize

        tangent -= dot(tangent, normal)[:, np.newaxis] * normal # Gram-Schmidt
        tangent /= np.sqrt(dot(tangent, tangent))[:, np.newaxis] # normalize

        bitangent -= dot(bitangent, normal)[:, np.newaxis] * normal # Gram-Schmidt
        bitangent -= dot(bitangent, tangent)[:, np.newaxis] * tangent # Gram-Schmidt
        bitangent /= np.sqrt(dot(bitangent, bitangent))[:, np.newaxis] # normalize

        return [tangent, bitangent, normal]

class Scene3D:
    def __init__(self, filename, rotation, translation):
        self.device = embree.Device()
        self.scene = self.device.make_scene()

        path = os.path.dirname(filename)
        gltf = GLTF2().load(filename)

        self.materials = []
        for gltf_material in gltf.materials:
            material = Material3D(path, gltf, gltf_material)
            self.materials.append(material)

        self.meshes = []
        for gltf_mesh in gltf.meshes:
            for primitive in gltf_mesh.primitives:
                mesh = Mesh3D(gltf, primitive, rotation, translation)
                self.meshes.append(mesh)

                geometry = self.device.make_geometry(embree.GeometryType.Triangle)

                vertex_buffer = geometry.set_new_buffer(embree.BufferType.Vertex, 0, embree.Format.Float3, 3 * np.dtype('float32').itemsize, mesh.position.shape[0])
                vertex_buffer[:] = mesh.position

                index_buffer = geometry.set_new_buffer(embree.BufferType.Index, 0, embree.Format.Uint3, 3 * np.dtype('uint32').itemsize, mesh.indices.shape[0])
                index_buffer[:] = mesh.indices

                geometry.commit()
                self.scene.attach_geometry(geometry)
                geometry.release()

        self.scene.commit()

    def rayCast(self, ray_origins, ray_directions, light_values, light_directions):
        n_rays = ray_origins.shape[0]

        ray_color = np.zeros((n_rays, 3), dtype=np.float32)

        rayhits = embree.RayHit1M(ray_origins.shape[0])
        rayhits.org[:] = ray_origins
        rayhits.dir[:] = ray_directions
        rayhits.tfar[:] = 0
        rayhits.flags[:] = 0

        context = embree.IntersectContext()

        active_rays = np.ones(n_rays, dtype=bool)

        while np.any(active_rays):
            rayhits.prim_id[active_rays] = embree.INVALID_GEOMETRY_ID
            rayhits.geom_id[active_rays] = embree.INVALID_GEOMETRY_ID
            rayhits.tnear[active_rays] = rayhits.tfar[active_rays] + 0.01
            rayhits.tfar[active_rays] = np.inf
            rayhits.tfar[~active_rays] = 0

            self.scene.intersect1M(context, rayhits)

            # Check for active rays that hit the scene.
            active_rays[active_rays] = (rayhits.geom_id[active_rays] != embree.INVALID_GEOMETRY_ID)

            for mesh_id in range(len(self.meshes)):
                mesh = self.meshes[mesh_id]
                material = self.materials[mesh.material]

                # Create mask with current mesh hits.
                mask = np.zeros(n_rays, dtype=bool)
                mask[active_rays] = (rayhits.geom_id[active_rays] == mesh_id)

                # Interpolate TexCoord vertex attribute.
                texcoord = mesh.interpolate_texcoord(rayhits.prim_id[mask], rayhits.uv[mask, :])

                # Fetch base color texture.
                base_color = material.fetch_base_color(texcoord)

                # Futher restrict mask to opaque hits.
                opaque = (base_color[..., 3] > 0.5)
                mask[mask] = opaque
                texcoord = texcoord[opaque]
                base_color = base_color[opaque, 0:3]

                # Interpolate TBN vertex attributes.
                tangent, bitangent, normal = mesh.interpolate_tbn(rayhits.prim_id[mask], rayhits.uv[mask, :])

                # Fetch normal map texture.
                normal_tanspace = material.fetch_normal(texcoord)
                normal_vector = normal_tanspace[..., 0:1] * tangent + normal_tanspace[..., 1:2] * bitangent + normal_tanspace[..., 2:3] * normal
                normal_vector /= np.sqrt(dot(normal_vector, normal_vector))[:, np.newaxis] # normalize

                # Fetch metallic roughness texture.
                metallic, roughness = material.fetch_metallic_roughness(texcoord)

                # Normalized view vector.
                view_vector = -ray_directions[mask]

                # Compute hit positions.
                position = rayhits.org[mask] + rayhits.dir[mask] * rayhits.tfar[mask, np.newaxis]

                for light_id in range(light_values.shape[0]):
                    # Normalized light vector.
                    light_vector = -light_directions[light_id]
                    
                    # Accumulate for each light.
                    ray_color[mask] +=  self.brdf_shading(view_vector, light_vector, normal_vector, metallic, roughness, base_color) * light_values[light_id]

                # Deactivate shaded rays.
                active_rays[mask] = False
        
        return ray_color
    
    def brdf_shading(self, view_vector, light_vector, normal_vector, metallic, roughness, base_color):
        alpha_sq = np.power(roughness, 4)

        halfway_vector = view_vector + light_vector
        halfway_vector /= np.sqrt(dot(halfway_vector, halfway_vector))[:, np.newaxis]

        NdotH = np.clip(dot(normal_vector, halfway_vector), 0.0, 1.0)[:, np.newaxis]
        VdotH = np.clip(dot(view_vector, halfway_vector), 0.0, 1.0)[:, np.newaxis]
        LdotN = np.clip(dot(light_vector, normal_vector), 0.0, 1.0)[:, np.newaxis]
        VdotN = np.clip(dot(view_vector, normal_vector), 0.0, 1.0)[:, np.newaxis]

        distribution = np.nan_to_num(alpha_sq / np.square(np.square(NdotH) * (alpha_sq - 1.0) + 1.0))

        ggx_v = LdotN * np.sqrt(np.square(VdotN) * (1.0 - alpha_sq) + alpha_sq)
        ggx_l = VdotN * np.sqrt(np.square(LdotN) * (1.0 - alpha_sq) + alpha_sq)
        ggx = ggx_v + ggx_l
        visibility = np.nan_to_num(0.5 / np.maximum(ggx, np.finfo(ggx.dtype).eps))

        dielectric_fresnel = 0.04 + (1.0 - 0.04) * np.power(1.0 - VdotH, 5)
        metal_fresnel = base_color + (1.0 - base_color) * np.power(1.0 - VdotH, 5)

        mixed_fresnel = (1.0 - metallic) * dielectric_fresnel + metallic * metal_fresnel

        difuse_color = (1.0 - mixed_fresnel) * (1.0 - metallic) * base_color
        specular_color = mixed_fresnel * distribution * visibility

        color = (difuse_color + specular_color) * LdotN

        return color
    
    def release(self):
        self.scene.release()
        self.device.release()
