import tkinter as tk
from tkinter import ttk

import numpy as np


def dot(a, b):
    return np.einsum('...i,...i', a, b)


def ray_sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius):
    center_vector = sphere_center - ray_origin
    t_c = dot(center_vector, ray_direction)

    delta = np.square(sphere_radius) + np.square(t_c) - dot(center_vector, center_vector)

    hit_mask = delta >= np.maximum(0.0, -t_c * np.abs(t_c))

    sqrt_delta = np.sqrt(delta[hit_mask])
    t = t_c[hit_mask] + np.where(t_c[hit_mask] >= sqrt_delta, -sqrt_delta, sqrt_delta)

    return [hit_mask, t]


def array2image(array):
    array = np.flip(np.swapaxes(array * 255.0, 0, 1), 0).astype(np.uint8)
    height, width = array.shape[:2]
    ppm_header = f'P6 {width} {height} 255 '.encode()
    data = ppm_header + array.tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')


class App:
    def __init__(self):
        self.mouse_x = 0
        self.mouse_y = 0

        self.res_x = 1024
        self.res_y = 1024
        self.fov_y = np.radians(60.0)

        self.pitch = 0.0
        self.yaw = 0.0

        self.root = tk.Tk()
        self.root.title("Shader")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        mainframe = ttk.Frame(self.root)
        mainframe.grid(column=0, row=0)

        self.viewport = ttk.Label(mainframe)
        self.viewport.grid(column=0, row=0)
        self.viewport.bind('<Button-1>',  self.leftClick)
        self.viewport.bind('<B1-Motion>', self.leftDrag)

        sideframe = ttk.Frame(mainframe, padding="10 10 10 10")
        sideframe.grid(column=1, row=0, sticky=tk.N)

        self.metallic_factor = tk.DoubleVar()
        self.roughness_factor = tk.DoubleVar()

        ttk.Label(sideframe, text='Metallic Factor', padding="4 4 4 4").grid(column=0, row=0, sticky='e')
        ttk.Label(sideframe, textvariable=self.metallic_factor).grid(column=2, row=0, sticky='w')
        metallic_scale = ttk.Scale(sideframe, orient='horizontal', length=200, from_=0.0, to=1.0, variable=self.metallic_factor, command=self.render)
        metallic_scale.grid(column=1, row=0)
        metallic_scale.set(1.0)

        ttk.Label(sideframe, text='Roughness Factor', padding="4 4 4 4").grid(column=0, row=1, sticky='e')
        ttk.Label(sideframe, textvariable=self.roughness_factor).grid(column=2, row=1, sticky='w')
        roughness_scale = ttk.Scale(sideframe, orient='horizontal', length=200, from_=0.0, to=1.0, variable=self.roughness_factor, command=self.render)
        roughness_scale.grid(column=1, row=1)
        roughness_scale.set(1.0)

        self.render()

    def run(self):
        self.root.mainloop()

    def view_matrix(self):
        return np.array([[np.cos(self.yaw), 0.0, -np.sin(self.yaw)],
                         [-np.sin(self.yaw) * np.sin(self.pitch), np.cos(self.pitch), -np.cos(self.yaw) * np.sin(self.pitch)],
                         [np.sin(self.yaw) * np.cos(self.pitch), np.sin(self.pitch), np.cos(self.yaw) * np.cos(self.pitch)]])

    def render(self, event=None):
        screen_x = np.linspace(-1.0, 1.0, num=self.res_x).reshape(self.res_x, 1) * np.tan(self.fov_y/2) * self.res_x/self.res_y
        screen_y = np.linspace(-1.0, 1.0, num=self.res_y).reshape(1, self.res_y) * np.tan(self.fov_y/2)

        ray_direction = np.stack(np.broadcast_arrays(screen_x, screen_y, -1.0), axis=-1)
        ray_direction /= np.sqrt(dot(ray_direction, ray_direction))[..., np.newaxis]

        ray_origin = np.array([0.0, 0.0, 2.2])

        view_matrix = self.view_matrix()
        ray_direction = ray_direction @ view_matrix
        ray_origin = ray_origin @ view_matrix

        sphere_center = np.array([0.0, 0.0, 0.0])
        sphere_radius = 1.0

        hit_mask, t = ray_sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius)

        p = ray_origin + ray_direction[hit_mask] * t[..., np.newaxis]

        view_vector = -ray_direction[hit_mask]

        normal_vector = p - sphere_center
        normal_vector /= np.sqrt(dot(normal_vector, normal_vector))[..., np.newaxis]

        light_vector = np.array([-0.5, 0.7, 0.5])
        light_vector /= np.sqrt(dot(light_vector, light_vector))

        halfway_vector = view_vector + light_vector
        halfway_vector /= np.sqrt(dot(halfway_vector, halfway_vector))[..., np.newaxis]

        NdotH = np.clip(dot(normal_vector, halfway_vector), 0.0, 1.0)[:, np.newaxis]
        LdotH = np.clip(dot(light_vector, halfway_vector), 0.0, 1.0)[:, np.newaxis]
        VdotH = np.clip(dot(view_vector, halfway_vector), 0.0, 1.0)[:, np.newaxis]
        LdotN = np.clip(dot(light_vector, normal_vector), 0.0, 1.0)[:, np.newaxis]
        VdotN = np.clip(dot(view_vector, normal_vector), 0.0, 1.0)[:, np.newaxis]

        alpha2 = np.square(self.roughness_factor.get())
        metallic = self.metallic_factor.get()
        base_color = np.array([1.0, 1.0, 1.0])

        distribution = (alpha2 / np.pi) / np.square(np.square(NdotH) * (alpha2 - 1.0) + 1.0)

        ggx_v = LdotN * np.sqrt(np.square(VdotN) * (1.0 - alpha2) + alpha2)
        ggx_l = VdotN * np.sqrt(np.square(LdotN) * (1.0 - alpha2) + alpha2)
        ggx_vl = ggx_v + ggx_l
        visibility = np.divide(0.5, ggx_vl, out=np.zeros_like(ggx_vl), where=~np.isclose(ggx_vl, 0.0))

        dielectric_fresnel = 0.04 + (1.0 - 0.04) * np.power(1.0 - VdotH, 5)
        metal_fresnel = base_color + (1.0 - base_color) * np.power(1.0 - VdotH, 5)

        mixed_fresnel = (1.0 - metallic) * dielectric_fresnel + metallic * metal_fresnel

        difuse_color = (1.0 - mixed_fresnel) * (1.0 - metallic) * base_color / np.pi
        specular_color = mixed_fresnel * distribution * visibility

        color = (difuse_color + specular_color) * LdotN

        u = np.arctan2(normal_vector[..., 0], -normal_vector[..., 2])/(2*np.pi) + 0.5
        v = normal_vector[..., 1]/2 + 0.5
        uv = np.stack((u, v), axis=-1)

        viewport_array = np.ones((self.res_x, self.res_y, 3)) * [48.0/255, 53.0/255, 66.0/255]
        viewport_array[hit_mask] = color

        self.viewport_image = array2image(viewport_array)
        self.viewport.configure(image=self.viewport_image)

    def leftClick(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

    def leftDrag(self, event):
        delta_x = event.x - self.mouse_x
        delta_y = event.y - self.mouse_y
        self.mouse_x = event.x
        self.mouse_y = event.y

        self.yaw = (self.yaw - delta_x * 0.01) % (2 * np.pi)
        self.pitch = np.clip(self.pitch + delta_y * 0.01, -np.pi/2, np.pi/2)

        self.render()


app = App()
app.run()