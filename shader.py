import tkinter as tk
from tkinter import filedialog, ttk

import cupy as cp
import numpy as np
from PIL import Image, ImageTk


def dot(a, b):
    return cp.einsum('...i,...i', a, b)


def ray_sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius):
    center_vector = sphere_center - ray_origin
    t_c = dot(center_vector, ray_direction)

    delta = cp.square(sphere_radius) + cp.square(t_c) - dot(center_vector, center_vector)

    hit_mask = delta >= cp.maximum(0.0, -t_c * cp.abs(t_c))

    sqrt_delta = cp.sqrt(delta[hit_mask])
    t = t_c[hit_mask] + cp.where(t_c[hit_mask] >= sqrt_delta, -sqrt_delta, sqrt_delta)

    return [hit_mask, t]


def array2image(array):
    array = cp.flip(cp.swapaxes(array * 255.0, 0, 1), 0).astype(cp.uint8)
    height, width = array.shape[:2]
    ppm_header = f'P6 {width} {height} 255 '.encode()
    data = ppm_header + array.get().tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')


class App:
    def __init__(self):
        self.mouse_x = 0
        self.mouse_y = 0

        self.res_x = 1024
        self.res_y = 1024
        self.fov_y = cp.radians(60.0)

        self.pitch = 0.0
        self.yaw = 0.0

        self.color_texture = cp.ones((256, 256, 3))
        self.normal_texture = cp.ones((256, 256, 3)) * cp.array([0.5, 0.5, 1.0])
        self.metallic_roughness_texture = cp.ones((256, 256, 3)) * cp.array([0.0, 1.0, 1.0])

        self.root = tk.Tk()
        self.root.title("PBR Shader")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        mainframe = ttk.Frame(self.root)
        mainframe.grid(column=0, row=0)

        self.viewport = ttk.Label(mainframe)
        self.viewport.grid(column=0, row=0)
        self.viewport.bind('<Button-1>',  self.leftClick)
        self.viewport.bind('<B1-Motion>', self.leftDrag)

        bottomframe = ttk.Frame(mainframe, padding="10 10 10 10")
        bottomframe.grid(column=0, row=1, sticky='s')

        self.minimum_value = tk.DoubleVar()
        self.average_value = tk.DoubleVar()
        self.maximum_value = tk.DoubleVar()

        ttk.Label(bottomframe, text='Minimum:', padding="4 4 4 4").grid(column=0, row=0, sticky='e')
        ttk.Label(bottomframe, textvariable=self.minimum_value).grid(column=1, row=0, sticky='w')
        ttk.Label(bottomframe, padding="150 4 4 4").grid(column=2, row=0)
        ttk.Label(bottomframe, text='Average:', padding="4 4 4 4").grid(column=3, row=0, sticky='e')
        ttk.Label(bottomframe, textvariable=self.average_value).grid(column=4, row=0, sticky='w')
        ttk.Label(bottomframe, padding="150 4 4 4").grid(column=5, row=0)
        ttk.Label(bottomframe, text='Maximum:', padding="4 4 4 4").grid(column=6, row=0, sticky='e')
        ttk.Label(bottomframe, textvariable=self.maximum_value).grid(column=7, row=0, sticky='w')

        sideframe = ttk.Frame(mainframe, padding="10 10 10 10")
        sideframe.grid(column=1, row=0, sticky='n')

        self.render_mode = tk.StringVar(value='PBR')
        self.metallic_factor = tk.DoubleVar(value=1.0)
        self.metallic_factor_text = tk.StringVar(value='{:.2f}'.format(self.metallic_factor.get()))
        self.roughness_factor = tk.DoubleVar(value=1.0)
        self.roughness_factor_text = tk.StringVar(value='{:.2f}'.format(self.roughness_factor.get()))
        self.clip_output = tk.BooleanVar(value=True)

        ttk.Label(sideframe, text='Render Mode:', padding="4 4 4 4").grid(column=0, row=0, sticky='e')
        render_mode_box = ttk.Combobox(sideframe, textvariable=self.render_mode)
        render_mode_box.grid(column=1, row=0, columnspan=2, sticky='ew')
        render_mode_box['values'] = ('Tangent', 'Bitangent', 'Normal', 'Texture UV', 'Color Texture', 'Normal Texture', 'Metallic', 'Roughness', '<L,N>',
                                     'Distribution', 'Visibility', 'Dielectric Fresnel', 'Metal Fresnel', 'Mixed Fresnel', 'Difuse', 'Specular', 'PBR')
        render_mode_box.state(['readonly'])
        render_mode_box.bind('<<ComboboxSelected>>', self.render)

        ttk.Label(sideframe, text='Metallic Factor:', padding="4 4 4 4").grid(column=0, row=1, sticky='e')
        metallic_scale = ttk.Scale(sideframe, orient='horizontal', length=200, from_=0.0, to=1.0, variable=self.metallic_factor, command=self.update_factors)
        metallic_scale.grid(column=1, row=1)
        ttk.Label(sideframe, textvariable=self.metallic_factor_text).grid(column=2, row=1, sticky='w')

        ttk.Label(sideframe, text='Roughness Factor:', padding="4 4 4 4").grid(column=0, row=2, sticky='e')
        roughness_scale = ttk.Scale(sideframe, orient='horizontal', length=200, from_=0.0, to=1.0, variable=self.roughness_factor, command=self.update_factors)
        roughness_scale.grid(column=1, row=2)
        ttk.Label(sideframe, textvariable=self.roughness_factor_text).grid(column=2, row=2, sticky='w')

        ttk.Checkbutton(sideframe, text='Clip Output', variable=self.clip_output, command=self.render).grid(column=0, row=3, columnspan=3)

        ttk.Label(sideframe, text='Color Texture:', padding="4 20 4 4").grid(column=0, row=4, columnspan=3)
        self.color_view = ttk.Label(sideframe)
        self.color_view.grid(column=0, row=5, columnspan=3)
        self.color_view.bind('<Button-1>',  self.load_color_texture)
        self.color_view.bind('<Button-3>',  self.reset_color_texture)
        self.color_image = array2image(self.color_texture)
        self.color_view.configure(image=self.color_image)

        ttk.Label(sideframe, text='Normal Texture:', padding="4 20 4 4").grid(column=0, row=6, columnspan=3)
        self.normal_view = ttk.Label(sideframe)
        self.normal_view.grid(column=0, row=7, columnspan=3)
        self.normal_view.bind('<Button-1>',  self.load_normal_texture)
        self.normal_view.bind('<Button-3>',  self.reset_normal_texture)
        self.normal_image = array2image(self.normal_texture)
        self.normal_view.configure(image=self.normal_image)

        ttk.Label(sideframe, text='Metallic-Roughness Texture:', padding="4 20 4 4").grid(column=0, row=8, columnspan=3)
        self.metallic_roughness_view = ttk.Label(sideframe)
        self.metallic_roughness_view.grid(column=0, row=9, columnspan=3)
        self.metallic_roughness_view.bind('<Button-1>',  self.load_metallic_roughness_texture)
        self.metallic_roughness_view.bind('<Button-3>',  self.reset_metallic_roughness_texture)
        self.metallic_roughness_image = array2image(self.metallic_roughness_texture)
        self.metallic_roughness_view.configure(image=self.metallic_roughness_image)

        self.render()

    def update_factors(self, event):
        self.metallic_factor_text.set('{:.2f}'.format(self.metallic_factor.get()))
        self.roughness_factor_text.set('{:.2f}'.format(self.roughness_factor.get()))
        self.render()

    def load_color_texture(self, event):
        filename = filedialog.askopenfilename()
        if len(filename) > 0:
            with Image.open(filename) as img:
                self.color_texture = cp.array(np.swapaxes(np.array(img.convert('RGBA')), 0, 1)[..., 0:3] / 255.0)
                self.color_image = ImageTk.PhotoImage(img.resize((256,256), Image.LANCZOS))
                self.color_view.configure(image=self.color_image)
                self.render()

    def reset_color_texture(self, event):
        self.color_texture = cp.ones((256, 256, 3))
        self.color_image = array2image(self.color_texture)
        self.color_view.configure(image=self.color_image)
        self.render()

    def load_normal_texture(self, event):
        filename = filedialog.askopenfilename()
        if len(filename) > 0:
            with Image.open(filename) as img:
                self.normal_texture = cp.array(np.swapaxes(np.array(img.convert('RGB')), 0, 1) / 255.0)
                self.normal_image = ImageTk.PhotoImage(img.resize((256,256), Image.LANCZOS))
                self.normal_view.configure(image=self.normal_image)
                self.render()

    def reset_normal_texture(self, event):
        self.normal_texture = cp.ones((256, 256, 3)) * cp.array([0.5, 0.5, 1.0])
        self.normal_image = array2image(self.normal_texture)
        self.normal_view.configure(image=self.normal_image)
        self.render()

    def load_metallic_roughness_texture(self, event):
        filename = filedialog.askopenfilename()
        if len(filename) > 0:
            with Image.open(filename) as img:
                self.metallic_roughness_texture = cp.array(np.swapaxes(np.array(img.convert('RGBA')), 0, 1)[..., 0:3] / 255.0)
                self.metallic_roughness_image = ImageTk.PhotoImage(img.resize((256,256), Image.LANCZOS))
                self.metallic_roughness_view.configure(image=self.metallic_roughness_image)
                self.render()

    def reset_metallic_roughness_texture(self, event):
        self.metallic_roughness_texture = cp.ones((256, 256, 3)) * cp.array([0.0, 1.0, 1.0])
        self.metallic_roughness_image = array2image(self.metallic_roughness_texture)
        self.metallic_roughness_view.configure(image=self.metallic_roughness_image)
        self.render()

    def run(self):
        self.root.mainloop()

    def update_viewport(self, values, mask):
        viewport_array = cp.ones((self.res_x, self.res_y, 3)) * cp.array([48.0/255, 53.0/255, 66.0/255])
        viewport_array[mask] = values

        self.viewport_image = array2image(viewport_array)
        self.viewport.configure(image=self.viewport_image)

    def update_stats(self, array):
        self.minimum_value.set(cp.amin(array))
        self.average_value.set(cp.average(array))
        self.maximum_value.set(cp.amax(array))

    def view_matrix(self):
        pitch_matrix = cp.array([[1.0, 0.0, 0.0],
                                 [0.0, np.cos(self.pitch), -np.sin(self.pitch)],
                                 [0.0, np.sin(self.pitch), np.cos(self.pitch)]])

        yaw_matrix = cp.array([[np.cos(self.yaw), 0.0, np.sin(self.yaw)],
                               [0.0, 1.0, 0.0],
                               [-np.sin(self.yaw), 0.0, np.cos(self.yaw)]])

        return pitch_matrix @ yaw_matrix

    def render(self, event=None):
        screen_x = cp.linspace(-1.0, 1.0, num=self.res_x).reshape(self.res_x, 1) * np.tan(self.fov_y/2) * self.res_x/self.res_y
        screen_y = cp.linspace(-1.0, 1.0, num=self.res_y).reshape(1, self.res_y) * np.tan(self.fov_y/2)

        ray_direction = cp.stack(cp.broadcast_arrays(screen_x, screen_y, cp.array(-1.0)), axis=-1)
        ray_direction /= cp.sqrt(dot(ray_direction, ray_direction))[..., cp.newaxis]

        ray_origin = cp.array([0.0, 0.0, 2.2])

        view_matrix = self.view_matrix()
        ray_direction = ray_direction @ view_matrix
        ray_origin = ray_origin @ view_matrix

        sphere_center = cp.array([0.0, 0.0, 0.0])
        sphere_radius = 1.0

        hit_mask, t = ray_sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius)

        position = ray_origin + ray_direction[hit_mask] * t[..., cp.newaxis]

        view_vector = -ray_direction[hit_mask]

        normal_vector = position - sphere_center
        normal_vector /= cp.sqrt(dot(normal_vector, normal_vector))[..., cp.newaxis]

        if self.render_mode.get() == 'Normal':
            self.update_viewport(normal_vector/2 + 0.5, hit_mask)
            self.update_stats(normal_vector)
            return

        tangent_vector = cp.array([0.0, 1.0, 0.0])
        tangent_vector = tangent_vector - dot(tangent_vector, normal_vector)[:, cp.newaxis] * normal_vector
        tangent_vector /= cp.sqrt(dot(tangent_vector, tangent_vector))[:, np.newaxis]

        if self.render_mode.get() == 'Tangent':
            self.update_viewport(tangent_vector/2 + 0.5, hit_mask)
            self.update_stats(tangent_vector)
            return

        bitangent_vector = cp.cross(normal_vector, tangent_vector)

        if self.render_mode.get() == 'Bitangent':
            self.update_viewport(bitangent_vector/2 + 0.5, hit_mask)
            self.update_stats(bitangent_vector)
            return

        u = cp.abs(cp.arctan2(normal_vector[..., 0], -normal_vector[..., 2])/cp.pi)
        v = 0.5 - normal_vector[..., 1]/2
        uv = cp.stack((u, v, cp.zeros_like(v)), axis=-1)

        if self.render_mode.get() == 'Texture UV':
            self.update_viewport(uv, hit_mask)
            self.update_stats(uv)
            return

        color_u = cp.minimum(u * self.color_texture.shape[0], self.color_texture.shape[0] - 1).astype(int)
        color_v = cp.minimum(v * self.color_texture.shape[1], self.color_texture.shape[1] - 1).astype(int)
        color_value = self.color_texture[color_u, color_v]

        if self.render_mode.get() == 'Color Texture':
            self.update_viewport(color_value, hit_mask)
            self.update_stats(color_value)
            return

        normal_u = cp.minimum(u * self.normal_texture.shape[0], self.normal_texture.shape[0] - 1).astype(int)
        normal_v = cp.minimum(v * self.normal_texture.shape[1], self.normal_texture.shape[1] - 1).astype(int)
        tbn = self.normal_texture[normal_u, normal_v]*2 - 1.0

        normal_vector = tbn[..., 0:1] * tangent_vector + tbn[..., 1:2] * bitangent_vector + tbn[..., 2:3] * normal_vector
        normal_vector /= cp.sqrt(dot(normal_vector, normal_vector))[:, cp.newaxis]

        if self.render_mode.get() == 'Normal Texture':
            self.update_viewport(normal_vector/2 + 0.5, hit_mask)
            self.update_stats(normal_vector)
            return

        metallic_roughness_u = cp.minimum(u * self.metallic_roughness_texture.shape[0], self.metallic_roughness_texture.shape[0] - 1).astype(int)
        metallic_roughness_v = cp.minimum(v * self.metallic_roughness_texture.shape[1], self.metallic_roughness_texture.shape[1] - 1).astype(int)
        metallic_roughness_value = self.metallic_roughness_texture[metallic_roughness_u, metallic_roughness_v]

        metallic_value = metallic_roughness_value[..., 2:3] * self.metallic_factor.get()
        roughness_value = metallic_roughness_value[..., 1:2] * self.roughness_factor.get()

        if self.render_mode.get() == 'Metallic':
            self.update_viewport(metallic_value, hit_mask)
            self.update_stats(metallic_value)
            return

        if self.render_mode.get() == 'Roughness':
            self.update_viewport(roughness_value, hit_mask)
            self.update_stats(roughness_value)
            return

        light_vector = cp.array([-0.5, 0.7, 0.5])
        light_vector /= cp.sqrt(dot(light_vector, light_vector))

        halfway_vector = view_vector + light_vector
        halfway_vector /= cp.sqrt(dot(halfway_vector, halfway_vector))[..., cp.newaxis]

        NdotH = cp.clip(dot(normal_vector, halfway_vector), 0.0, 1.0)[:, cp.newaxis]
        LdotH = cp.clip(dot(light_vector, halfway_vector), 0.0, 1.0)[:, cp.newaxis]
        VdotH = cp.clip(dot(view_vector, halfway_vector), 0.0, 1.0)[:, cp.newaxis]
        LdotN = cp.clip(dot(light_vector, normal_vector), 0.0, 1.0)[:, cp.newaxis]
        VdotN = cp.clip(dot(view_vector, normal_vector), 0.0, 1.0)[:, cp.newaxis]

        if self.render_mode.get() == '<L,N>':
            self.update_viewport(LdotN, hit_mask)
            self.update_stats(LdotN)
            return

        alpha_sq = np.power(roughness_value, 4)

        distribution = cp.nan_to_num(alpha_sq / cp.square(cp.square(NdotH) * (alpha_sq - 1.0) + 1.0))

        if self.render_mode.get() == 'Distribution':
            self.update_viewport(cp.clip(distribution, 0.0, 1.0) if self.clip_output.get() else distribution, hit_mask)
            self.update_stats(distribution)
            return

        ggx_v = LdotN * cp.sqrt(cp.square(VdotN) * (1.0 - alpha_sq) + alpha_sq)
        ggx_l = VdotN * cp.sqrt(cp.square(LdotN) * (1.0 - alpha_sq) + alpha_sq)
        visibility = cp.nan_to_num(0.5 / (ggx_v + ggx_l))

        if self.render_mode.get() == 'Visibility':
            self.update_viewport(cp.clip(visibility, 0.0, 1.0) if self.clip_output.get() else visibility, hit_mask)
            self.update_stats(visibility)
            return

        dielectric_fresnel = 0.04 + (1.0 - 0.04) * cp.power(1.0 - VdotH, 5)

        if self.render_mode.get() == 'Dielectric Fresnel':
            self.update_viewport(dielectric_fresnel, hit_mask)
            self.update_stats(dielectric_fresnel)
            return

        metal_fresnel = color_value + (1.0 - color_value) * cp.power(1.0 - VdotH, 5)

        if self.render_mode.get() == 'Metal Fresnel':
            self.update_viewport(metal_fresnel, hit_mask)
            self.update_stats(metal_fresnel)
            return

        mixed_fresnel = (1.0 - metallic_value) * dielectric_fresnel + metallic_value * metal_fresnel

        if self.render_mode.get() == 'Mixed Fresnel':
            self.update_viewport(mixed_fresnel, hit_mask)
            self.update_stats(mixed_fresnel)
            return

        difuse_value = (1.0 - mixed_fresnel) * (1.0 - metallic_value) * color_value

        if self.render_mode.get() == 'Difuse':
            self.update_viewport(difuse_value, hit_mask)
            self.update_stats(difuse_value)
            return

        specular_value = mixed_fresnel * distribution * visibility

        if self.render_mode.get() == 'Specular':
            self.update_viewport(cp.clip(specular_value, 0.0, 1.0) if self.clip_output.get() else specular_value, hit_mask)
            self.update_stats(specular_value)
            return

        pbr_value = (difuse_value + specular_value) * LdotN

        if self.render_mode.get() == 'PBR':
            self.update_viewport(cp.clip(pbr_value, 0.0, 1.0) if self.clip_output.get() else pbr_value, hit_mask)
            self.update_stats(pbr_value)
            return

    def leftClick(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

        self.render(event)

    def leftDrag(self, event):
        delta_x = event.x - self.mouse_x
        delta_y = event.y - self.mouse_y
        self.mouse_x = event.x
        self.mouse_y = event.y

        self.yaw = (self.yaw + delta_x * 0.01) % (2 * np.pi)
        self.pitch = np.clip(self.pitch + delta_y * 0.01, -np.pi/3, np.pi/3)

        self.render(event)


app = App()
app.run()