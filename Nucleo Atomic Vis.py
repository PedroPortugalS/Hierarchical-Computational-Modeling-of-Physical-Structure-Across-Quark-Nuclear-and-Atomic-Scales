
import numpy as np
from scipy.special import sph_harm, genlaguerre, factorial
from scipy.ndimage import gaussian_filter
from skimage import measure
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox,
                             QCheckBox, QScrollArea, QSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import pyqtgraph.opengl as gl
import pyqtgraph as pg

# Physical constants
BOHR_RADIUS = 0.529177  # Angstroms
NUCLEAR_RADIUS_COEFFICIENT = 1.2  # fm

# Liquid Drop Model coefficients (MeV)
a_v = 15.8   # Volume
a_s = 18.3   # Surface
a_c = 0.714  # Coulomb
a_a = 23.2   # Asymmetry
a_p = 12.0   # Pairing

class NuclearPhysics:
    """Liquid Drop Model"""
    
    @staticmethod
    def pairing_term(A, Z):
        N = A - Z
        if A % 2 == 1:
            return 0
        elif Z % 2 == 0 and N % 2 == 0:
            return +a_p / (A ** 0.5)
        else:
            return -a_p / (A ** 0.5)
    
    @staticmethod
    def binding_energy(A, Z):
        # Semi-empirical mass formula breaks down for very light nuclei
        if A < 4:
            # Use experimental values for deuteron and He-3
            experimental_BE = {1: 0.0, 2: 2.224, 3: 7.718}
            return experimental_BE.get(A, 0.0)
        
        N = A - Z
        volume = a_v * A
        surface = a_s * (A ** (2/3))
        coulomb = a_c * (Z * (Z - 1)) / (A ** (1/3))
        asymmetry = a_a * ((A - 2*Z) ** 2) / A
        delta = NuclearPhysics.pairing_term(A, Z)
        return volume - surface - coulomb - asymmetry + delta

    @staticmethod
    def binding_energy_per_nucleon(A, Z):
        return NuclearPhysics.binding_energy(A, Z) / A if A > 0 else 0
    
    @staticmethod
    def nuclear_radius(A):
        if A <= 0:
            return 0.0
        return NUCLEAR_RADIUS_COEFFICIENT * (A ** (1/3))
    
class QuantumVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.Z = 1
        self.N = 0
        self.A = 1
        
        self.threshold = 0.15
        self.resolution = 40
        self.show_nucleus = True
        self.show_quarks = True
        self.nucleus_scale = 1000.0
        
        # Camera distance tracking for label visibility
        self.camera_distance = 5.0
        self.quark_label_distance_threshold = 0.5  # Show labels when closer than 
        
        self.colors = {
            'blue': (0.3, 0.6, 1.0, 0.5),
            'purple': (0.7, 0.3, 1.0, 0.5),
            'cyan': (0.3, 1.0, 0.9, 0.5),
            'yellow': (1.0, 0.9, 0.2, 0.5),
            'green': (0.3, 1.0, 0.3, 0.5),
            'orange': (1.0, 0.6, 0.2, 0.5),
            'pink': (1.0, 0.4, 0.7, 0.5),
            'red': (1.0, 0.3, 0.3, 0.5)
        }
        
        self.color_list = ['blue', 'cyan', 'purple', 'green', 'yellow', 'orange', 'pink', 'red']
        
        self.electron_config = []
        self.orbital_checkboxes = {}
        self.visible_orbitals = set()
        
        self.volume_cache = {}
        self.mesh_cache = {}
        self.nucleus_items = []
        self.quark_labels = []  # Track quark labels separately for visibility control
        
        # Animation for orbital "wiggle"
        self.animation_time = 0.0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_orbitals)
        self.animation_timer.start(30)  # Update every 30ms for smoother animation (~33 FPS)
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.setSingleShot(True)
        
        
        # Timer for camera distance updates
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_distance)
        self.camera_timer.start(100)  # Update every 100ms
        
        self.init_ui()
        self.update_nucleus_info()
        self.update_electron_config()
        self.update_visualization()
        
    def update_camera_distance(self):
        """Track camera distance and update quark label visibility"""
        camera_params = self.view.cameraParams()
        new_distance = camera_params.get('distance', 5.0)
        
        if abs(new_distance - self.camera_distance) > 0.1:
            self.camera_distance = new_distance
            self.update_quark_label_visibility()
    
    def update_quark_label_visibility(self):
        """Show/hide quark labels based on camera distance"""
        show_labels = self.camera_distance < self.quark_label_distance_threshold
        
        for label in self.quark_labels:
            label.setVisible(show_labels)
    
    def animate_orbitals(self):
        """Animate orbital meshes with quantum fluctuations - fluid cloud-like behavior"""
        self.animation_time += 0.085
        
        for key in self.visible_orbitals:
            if key in self.mesh_cache:
                n, l = key
                
                frequency = 0.67 + n * 0.135 + l * 0.10
                phase = l * 0.5  
                
                wave1 = np.sin(self.animation_time * frequency + phase)
                wave2 = np.sin(self.animation_time * frequency * 1.3 + phase + 1.5)
                wave3 = np.cos(self.animation_time * frequency * 0.7 + phase * 0.5)
                
                scale = 1.0 + 0.05 * wave1 + 0.027 * wave2 + 0.0235 * wave3
                rotation_angle = self.animation_time * 0.27 * (1 + l * 0.10)
                
                for mesh_idx, mesh in enumerate(self.mesh_cache[key]):
                    if mesh.visible():
                        mesh.resetTransform()
                        mesh.scale(scale, scale, scale)
                        layer_phase = mesh_idx * 0.05
                        mesh.rotate(rotation_angle + layer_phase, 0, 0, 1)
        
    def get_electron_configuration(self, Z):
        orbital_order = [
            (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2),
            (4, 1), (5, 0), (4, 2), (5, 1), (6, 0), (4, 3), (5, 2),
            (6, 1), (7, 0), (5, 3), (6, 2), (7, 1)
        ]
        
        max_electrons = {0: 2, 1: 6, 2: 10, 3: 14}
        
        config = []
        electrons_left = Z
        
        for n, l in orbital_order:
            if electrons_left <= 0:
                break
            
            max_e = max_electrons[l]
            electrons_in_orbital = min(electrons_left, max_e)
            
            # Store as subshell (n, l) with total electron count
            config.append((n, l, electrons_in_orbital))
            electrons_left -= electrons_in_orbital
        
        return config

    def apply_electron_configuration_exceptions(self, config, Z):
        """
        Apply known exceptions to Aufbau principle filling.
        """
        exceptions = {
            24: ('3d', '4s', 5, 1),  # Cr: 3d5 4s1
            29: ('3d', '4s', 10, 1), # Cu: 3d10 4s1
            41: ('4d', '5s', 4, 1),  # Nb: 4d4 5s1
            42: ('4d', '5s', 5, 1),  # Mo: 4d5 5s1
            44: ('4d', '5s', 7, 1),  # Ru: 4d7 5s1
            45: ('4d', '5s', 8, 1),  # Rh: 4d8 5s1
            46: ('4d', '5s', 10, 0), # Pd: 4d10 5s0
            47: ('4d', '5s', 10, 1), # Ag: 4d10 5s1
            78: ('5d', '6s', 9, 1),  # Pt: 5d9 6s1
            79: ('5d', '6s', 10, 1), # Au: 5d10 6s1
        }
        
        if Z not in exceptions:
            return config
        
        d_orbital_name, s_orbital_name, target_d_electrons, target_s_electrons = exceptions[Z]
        
        d_n = int(d_orbital_name[0])
        d_l = 2
        s_n = int(s_orbital_name[0])
        s_l = 0
        
        new_config = []
        for n, l, count in config:
            if n == d_n and l == d_l:
                if target_d_electrons > 0:
                    new_config.append((n, l, target_d_electrons))
            elif n == s_n and l == s_l:
                if target_s_electrons > 0:
                    new_config.append((n, l, target_s_electrons))
            else:
                new_config.append((n, l, count))
        
        return new_config
        

    def get_Z_eff(self, Z, n, l):
        """Slater's rules for effective nuclear charge"""
        # Count electrons in groups
        electrons_by_group = {}
        for conf_n, conf_l, count in self.electron_config:
            # Group by (n, l type)
            if conf_n < n or (conf_n == n and conf_l < l):
                key = (conf_n, conf_l)
                electrons_by_group[key] = electrons_by_group.get(key, 0) + count
        
        shielding = 0
        
        # Same group (n,l): 0.35 per electron
        for conf_n, conf_l, count in self.electron_config:
            if conf_n == n and conf_l == l:
                shielding += 0.35 * (count - 1)  # Exclude the electron itself
        
        # n-1 group: 0.85 per electron
        for (conf_n, conf_l), count in electrons_by_group.items():
            if conf_n == n - 1:
                shielding += 0.85 * count
        
        # n-2 and below: 1.00 per electron
        for (conf_n, conf_l), count in electrons_by_group.items():
            if conf_n < n - 1:
                shielding += 1.00 * count
        
        return max(Z - shielding, 1.0)
    
    def radial_wavefunction(self, r, n, l, Z_eff):
        a0 = BOHR_RADIUS
        rho = 2 * Z_eff * r / (n * a0)
        
        norm = np.sqrt(
            (2 * Z_eff / (n * a0))**3 * 
            factorial(n - l - 1) / (2 * n * factorial(n + l))
        )
        
        laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)
        R = norm * (rho ** l) * np.exp(-rho / 2) * laguerre_poly(rho)
        return R
    
    def probability_density(self, r, theta, phi, n, l, m, Z_eff):
        """Calculate probability density for a single (n,l,m) orbital"""
        R = self.radial_wavefunction(r, n, l, Z_eff)
        Y = sph_harm(m, l, phi, theta)
        prob = r**2 * np.abs(R)**2 * np.abs(Y)**2
        return prob
    
    def generate_volume_grid(self, n, l, m, Z, resolution):
        Z_eff = self.get_Z_eff(Z, n, l)
        # Orbital size scales as n²/Z_eff (Bohr model)
        max_radius = (n**2 / Z_eff) * BOHR_RADIUS * 4.0
        
        # Create grid centered at origin with proper spacing
        spacing = 2 * max_radius / (resolution - 1)
        x = np.linspace(-max_radius, max_radius, resolution)
        y = np.linspace(-max_radius, max_radius, resolution)
        z = np.linspace(-max_radius, max_radius, resolution)
        
        X, Y, Z_grid = np.meshgrid(x, y, z, indexing='ij')
        
        R = np.sqrt(X**2 + Y**2 + Z_grid**2)
        R[R < 0.01] = 0.01
        
        Theta = np.arccos(Z_grid / R)
        Phi = np.arctan2(Y, X)
        
        volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    r = R[i, j, k]
                    theta = Theta[i, j, k]
                    phi = Phi[i, j, k]
                    volume[i, j, k] = self.probability_density(r, theta, phi, n, l, m, Z_eff)
        
        max_val = np.max(volume)
        if max_val > 0:
            volume /= max_val
        
        volume = gaussian_filter(volume, sigma=1.5)
        
        max_val = np.max(volume)
        if max_val > 0:
            volume /= max_val
        
        return volume, max_radius
    
    def draw_nucleus(self):
        """Draw nucleus with nucleons showing quark structure and residual strong force"""
        # Clear previous nucleus items
        for item in self.nucleus_items:
            self.view.removeItem(item)
        self.nucleus_items = []
        self.quark_labels = []
        
        if not self.show_nucleus or self.A == 0:
            return
        
        nuclear_radius_fm = NuclearPhysics.nuclear_radius(self.A)
        nuclear_radius_angstrom = nuclear_radius_fm * 1e-5
        display_radius = nuclear_radius_angstrom * self.nucleus_scale
        
        nucleon_radius = display_radius * 0.15
        
        num_shells = max(1, int(np.ceil(self.A ** (1/3))))
        nucleons_per_shell = []
        remaining = self.A
        
        for shell in range(num_shells):
            shell_nucleons = min(remaining, 4 * (shell + 1) ** 2)
            nucleons_per_shell.append(shell_nucleons)
            remaining -= shell_nucleons
        
        proton_count = 0
        neutron_count = 0
        
        # Store nucleon positions for binding force visualization
        nucleon_positions = []
        nucleon_types = []  # True for proton, False for neutron
        
        for shell_idx, n_nucleons in enumerate(nucleons_per_shell):
            shell_radius = display_radius * (0.3 + 0.7 * (shell_idx + 1) / num_shells)
            
            phi = np.pi * (3. - np.sqrt(5.))
            
            for i in range(n_nucleons):
                y = 1 - (i / float(n_nucleons - 1)) * 2 if n_nucleons > 1 else 0
                radius_at_y = np.sqrt(1 - y * y)
                
                theta = phi * i
                
                x = np.cos(theta) * radius_at_y
                z = np.sin(theta) * radius_at_y
                
                pos = np.array([x, y, z]) * shell_radius
                
                is_proton = proton_count < self.Z
                
                nucleon_positions.append(pos)
                nucleon_types.append(is_proton)
                
                if self.show_quarks:
                    self.draw_nucleon_with_quarks(pos, nucleon_radius, is_proton)
                else:
                    if is_proton:
                        color = (1.0, 0.2, 0.2, 0.9)
                        proton_count += 1
                    else:
                        color = (0.2, 0.4, 1.0, 0.9)
                        neutron_count += 1
                    
                    md = gl.MeshData.sphere(rows=10, cols=10, radius=nucleon_radius)
                    nucleon = gl.GLMeshItem(
                        meshdata=md,
                        color=color,
                        smooth=True,
                        drawEdges=False,
                        glOptions='translucent'
                    )
                    nucleon.translate(pos[0], pos[1], pos[2])
                    self.view.addItem(nucleon)
                    self.nucleus_items.append(nucleon)
                
                if is_proton:
                    proton_count += 1
                else:
                    neutron_count += 1
        
        # Draw residual strong force (nuclear binding) between nearby nucleons
        if len(nucleon_positions) > 1:
            self.draw_nuclear_binding_forces(nucleon_positions, nucleon_radius)
        
        # Nuclear boundary sphere (transparent)
        md = gl.MeshData.sphere(rows=20, cols=20, radius=display_radius)
        boundary = gl.GLMeshItem(
            meshdata=md,
            color=(1.0, 1.0, 0.3, 0.1),
            smooth=True,
            drawEdges=True,
            glOptions='translucent'
        )
        self.view.addItem(boundary)
        self.nucleus_items.append(boundary)
        
        # Update label visibility based on current camera distance
        self.update_quark_label_visibility()
    
    def draw_nuclear_binding_forces(self, positions, nucleon_radius):
        """
        Draw residual strong force connections between nucleons
        Following Liquid Drop Model: nucleons interact with nearby neighbors
        The strong force has short range (~2-3 fm)
        
        In the Liquid Drop Model, nucleons behave like molecules in a liquid:
        - Every nucleon interacts with ALL nearby neighbors within range
        - This saturates at about 2-3 fm (the strong force range)
        - Each connection represents the residual strong force (pion exchange)
        """
        # Maximum interaction distance (saturation of nuclear force)
        # This is approximately 2-3 nucleon diameters in the strong force range
        max_distance = nucleon_radius * 3.5  # Increased range to show more connections
        
        connections_drawn = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                
                # Draw connection if within nuclear force range
                if dist < max_distance:
                    # Strength decreases with distance (Yukawa-like potential)
                    strength = np.exp(-dist / (max_distance * 0.4))  # Exponential falloff
                    
                    # Only draw if strength is significant
                    if strength > 0.1:
                        line_radius = nucleon_radius * 0.10 * strength
                        self.draw_nuclear_force_line(positions[i], positions[j], line_radius, strength)
                        connections_drawn += 1
        
        print(f"  Nuclear binding: {connections_drawn} force connections between {len(positions)} nucleons")
    
    def draw_nuclear_force_line(self, pos1, pos2, radius, strength):
        """
        Draw residual strong force (nuclear force) between nucleons
        Mediated by pion exchange - binds nucleons in nucleus
        """
        direction = pos2 - pos1
        length = np.linalg.norm(direction)
        
        if length < 0.001:
            return
        
        num_segments = 6
        vertices = []
        faces = []
        colors = []
        
        direction_norm = direction / length
        
        # Create perpendicular basis
        if abs(direction_norm[2]) < 0.9:
            perp1 = np.cross(direction_norm, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction_norm, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction_norm, perp1)
        
        # Create tube with variable thickness
        num_points = 8
        for i in range(num_points + 1):
            t = i / num_points
            center = pos1 + direction * t
            
            # Narrowing at ends (field line effect)
            local_radius = radius * (0.7 + 0.3 * (1 - abs(2 * t - 1)))
            
            for j in range(num_segments):
                angle = 2 * np.pi * j / num_segments
                offset = local_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                
                vertices.append(center + offset)
                
                # Orange/amber for nuclear force (pion exchange)
                base_alpha = 0.35 * strength
                alpha = base_alpha * (1 - 0.3 * abs(2 * t - 1))
                colors.append([1.0, 0.65, 0.15, alpha])
        
        # Create faces
        for i in range(num_points):
            for j in range(num_segments):
                next_j = (j + 1) % num_segments
                
                v1 = i * num_segments + j
                v2 = i * num_segments + next_j
                v3 = (i + 1) * num_segments + next_j
                v4 = (i + 1) * num_segments + j
                
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        vertices = np.array(vertices)
        faces = np.array(faces)
        colors = np.array(colors)
        
        force_mesh = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            vertexColors=colors,
            smooth=True,
            drawEdges=False,
            glOptions='translucent'
        )
        self.view.addItem(force_mesh)
        self.nucleus_items.append(force_mesh)
    
    def draw_nucleon_with_quarks(self, center_pos, nucleon_radius, is_proton):
        """Draw nucleon showing UUD (proton) or UDD (neutron) quark structure with distance-based labels"""
        quark_radius = nucleon_radius * 0.22
        quark_distance = nucleon_radius * 0.45
        
        # Position quarks in equilateral triangle
        angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
        quark_positions = []
        
        for angle in angles:
            qx = center_pos[0] + quark_distance * np.cos(angle)
            qy = center_pos[1] + quark_distance * np.sin(angle)
            qz = center_pos[2]
            quark_positions.append(np.array([qx, qy, qz]))
        
        # Quark colors representing QCD color charge
        if is_proton:
            # Proton = UUD (Up, Up, Down)
            quark_colors = [
                (0.2, 0.4, 1.0, 1.0),      # U - Blue
                (1.0, 0.0, 0.0, 1.0),      # U - Red
                (0.0, 1.0, 0.0, 1.0),      # D - Green
            ]
            quark_types = ['U', 'U', 'D']
            nucleon_color = (1.0, 0.2, 0.2, 0.25)
        else:
            # Neutron = UDD (Up, Down, Down)
            quark_colors = [
                (0.2, 0.4, 1.0, 1.0),      # U - Blue
                (0.0, 1.0, 0.0, 1.0),      # D - Green
                (1.0, 0.0, 0.0, 1.0),      # D - Red
            ]
            quark_types = ['U', 'D', 'D']
            nucleon_color = (0.2, 0.4, 1.0, 0.25)
        
        # Draw quarks as spheres with distance-based text labels
        for i, (pos, color, q_type) in enumerate(zip(quark_positions, quark_colors, quark_types)):
            md = gl.MeshData.sphere(rows=10, cols=10, radius=quark_radius)
            quark = gl.GLMeshItem(
                meshdata=md,
                color=color,
                smooth=True,
                drawEdges=False,
                glOptions='opaque'
            )
            quark.translate(pos[0], pos[1], pos[2])
            self.view.addItem(quark)
            self.nucleus_items.append(quark)
            
            # Add text label directly touching the top surface of the quark
            # Position it at quark radius distance (touching the surface)
            label_pos = pos + np.array([0, 0, quark_radius * 0.1])  # Barely above surface
            text = gl.GLTextItem(
                pos=label_pos,
                text=q_type,
                color=(255, 255, 255, 240),
                font=QFont("Arial", 10, QFont.Bold)
            )
            # Set depth test so labels are occluded by objects in front
            text.setGLOptions('opaque')  # Use opaque rendering with depth testing
            self.view.addItem(text)
            self.nucleus_items.append(text)
            self.quark_labels.append(text)  # Track for visibility control
        
        # Draw gluon flux tubes connecting all three quarks
        flux_pairs = [(0, 1), (1, 2), (2, 0)]
        
        for idx1, idx2 in flux_pairs:
            pos1 = quark_positions[idx1]
            pos2 = quark_positions[idx2]
            color1 = quark_colors[idx1]
            color2 = quark_colors[idx2]
            
            self.draw_gluon_flux_tube(pos1, pos2, quark_radius * 0.25, color1, color2)
        
        # Nucleon boundary
        md = gl.MeshData.sphere(rows=12, cols=12, radius=nucleon_radius)
        boundary = gl.GLMeshItem(
            meshdata=md,
            color=nucleon_color,
            smooth=True,
            drawEdges=True,
            glOptions='translucent'
        )
        boundary.translate(center_pos[0], center_pos[1], center_pos[2])
        self.view.addItem(boundary)
        self.nucleus_items.append(boundary)
    
    def draw_gluon_flux_tube(self, pos1, pos2, radius, color1, color2):
        """Draw gluon flux tube with spring-like structure and color mixing"""
        direction = pos2 - pos1
        length = np.linalg.norm(direction)
        
        if length < 0.001:
            return
        
        num_segments = 8
        num_coils = 3
        
        vertices = []
        faces = []
        colors = []
        
        direction_norm = direction / length
        
        # Create perpendicular basis
        if abs(direction_norm[2]) < 0.9:
            perp1 = np.cross(direction_norm, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction_norm, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction_norm, perp1)
        
        # Generate spring coil vertices
        num_points = 16
        for i in range(num_points + 1):
            t = i / num_points
            
            # Position along axis
            center = pos1 + direction * t
            
            # Spiral motion around axis
            angle = t * num_coils * 2 * np.pi
            spiral_radius = radius * 1.5
            
            # Offset perpendicular to direction
            offset = spiral_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            
            # Create circle of vertices around this point
            for j in range(num_segments):
                circle_angle = 2 * np.pi * j / num_segments
                tube_offset = radius * 0.6 * (
                    np.cos(circle_angle) * perp1 + 
                    np.sin(circle_angle) * perp2
                )
                
                vertex = center + offset + tube_offset
                vertices.append(vertex)
                
                # Color interpolation between quarks with gluon rainbow effect
                color_t = (t + np.sin(angle * 2) * 0.2) % 1.0
                r = color1[0] * (1 - color_t) + color2[0] * color_t
                g = color1[1] * (1 - color_t) + color2[1] * color_t
                b = color1[2] * (1 - color_t) + color2[2] * color_t
                
                # Add chromatic variation
                phase = (angle + circle_angle) % (2 * np.pi)
                brightness = 0.7 + 0.3 * np.sin(phase * 3)
                
                colors.append([r * brightness, g * brightness, b * brightness, 0.8])
        
        # Create faces
        for i in range(num_points):
            for j in range(num_segments):
                next_j = (j + 1) % num_segments
                
                v1 = i * num_segments + j
                v2 = i * num_segments + next_j
                v3 = (i + 1) * num_segments + next_j
                v4 = (i + 1) * num_segments + j
                
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        vertices = np.array(vertices)
        faces = np.array(faces)
        colors = np.array(colors)
        
        flux_mesh = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            vertexColors=colors,
            smooth=True,
            drawEdges=False,
            glOptions='translucent'
        )
        self.view.addItem(flux_mesh)
        self.nucleus_items.append(flux_mesh)
    
    def init_ui(self):
        self.setWindowTitle("Complete Atomic Visualizer - Quark Structure + Nuclear Binding")
        self.setGeometry(100, 100, 1600, 950)
        self.setStyleSheet("background-color: #0d0d19; color: white;")
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        left_sidebar = self.create_left_sidebar()
        layout.addWidget(left_sidebar)
        
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('#0d0d19')
        self.view.setCameraPosition(distance=5)
        layout.addWidget(self.view, stretch=1)
        
        right_sidebar = self.create_right_sidebar()
        layout.addWidget(right_sidebar)
        
        grid = gl.GLGridItem()
        grid.setSize(40, 40)
        grid.setSpacing(2, 2)
        grid.translate(0, 0, 0)
        self.view.addItem(grid)
        
        axis = gl.GLAxisItem()
        axis.setSize(3, 3, 3)
        self.view.addItem(axis)
    
    def create_left_sidebar(self):
        sidebar = QWidget()
        sidebar.setMinimumWidth(240)
        sidebar.setMaximumWidth(240)
        sidebar.setStyleSheet("background-color: #0f0f19;")
        layout = QVBoxLayout(sidebar)
        
        self.title_label = QLabel("Atomic Structure")
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.title_label.setStyleSheet("color: #667eea;")
        layout.addWidget(self.title_label)
        
        self.config_label = QLabel("H: 1s¹")
        self.config_label.setStyleSheet("color: #c8c8c8; font-size: 13px;")
        self.config_label.setWordWrap(True)
        layout.addWidget(self.config_label)
        
        self.nuclear_label = QLabel("")
        self.nuclear_label.setStyleSheet("""
            color: #ffa726;
            font-size: 11px;
            padding: 8px;
            background-color: rgba(255, 167, 38, 0.1);
            border-radius: 5px;
        """)
        self.nuclear_label.setWordWrap(True)
        layout.addWidget(self.nuclear_label)
        
        self.loading_label = QLabel("")
        self.loading_label.setStyleSheet("""
            color: #667eea;
            font-size: 14px;
            font-weight: bold;
            padding: 8px;
            background-color: rgba(102, 126, 234, 0.2);
            border-radius: 5px;
        """)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.hide()
        layout.addWidget(self.loading_label)
        
        self.loading_spinner_angle = 0
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self.update_loading_spinner)
        self.loading_timer.setInterval(100)
        
        layout.addSpacing(15)
        
        nuclear_group = QGroupBox("NUCLEAR CONFIG")
        nuclear_group.setStyleSheet("QGroupBox { color: #ffa726; font-weight: bold; }")
        nuclear_layout = QVBoxLayout()
        
        self.Z_slider = self.create_slider("Protons (Z)", 1, 118, 1, nuclear_layout)
        self.Z_slider.valueChanged.connect(self.on_z_changed)
        
        neutron_label = QLabel("Neutrons (N)")
        neutron_label.setStyleSheet("color: #ffa726;")
        nuclear_layout.addWidget(neutron_label)
        
        neutron_hlayout = QHBoxLayout()
        self.neutron_spinbox = QSpinBox()
        self.neutron_spinbox.setMinimum(0)
        self.neutron_spinbox.setMaximum(200)
        self.neutron_spinbox.setValue(0)
        self.neutron_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #323246;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        self.neutron_spinbox.valueChanged.connect(self.on_neutron_changed)
        
        auto_neutron_btn = QPushButton("Auto")
        auto_neutron_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffa726;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #ffb74d;
            }
        """)
        auto_neutron_btn.clicked.connect(self.auto_set_neutrons)
        
        neutron_hlayout.addWidget(self.neutron_spinbox)
        neutron_hlayout.addWidget(auto_neutron_btn)
        nuclear_layout.addLayout(neutron_hlayout)
        
        nuclear_group.setLayout(nuclear_layout)
        layout.addWidget(nuclear_group)
        
        layout.addSpacing(15)
        
        vis_group = QGroupBox("VISUALIZATION")
        vis_group.setStyleSheet("QGroupBox { color: #667eea; font-weight: bold; }")
        vis_layout = QVBoxLayout()
        
        self.nucleus_checkbox = QCheckBox("Show Nucleus")
        self.nucleus_checkbox.setChecked(True)
        self.nucleus_checkbox.setStyleSheet("color: white;")
        self.nucleus_checkbox.stateChanged.connect(self.toggle_nucleus)
        vis_layout.addWidget(self.nucleus_checkbox)
        
        self.quark_checkbox = QCheckBox("Show Quark Structure")
        self.quark_checkbox.setChecked(True)
        self.quark_checkbox.setStyleSheet("color: white;")
        self.quark_checkbox.stateChanged.connect(self.toggle_quarks)
        vis_layout.addWidget(self.quark_checkbox)
        
        self.threshold_slider = self.create_slider("Orbital Threshold", 5, 50, 15, vis_layout, scale=100)
        self.resolution_slider = self.create_slider("Resolution", 30, 70, 40, vis_layout)
        
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)
        
        layout.addSpacing(15)
        
        update_btn = QPushButton("Update Visualization")
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7c94f5;
            }
        """)
        update_btn.clicked.connect(self.update_visualization)
        layout.addWidget(update_btn)
        
        layout.addSpacing(10)
        
        exit_btn = QPushButton("EXIT")
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)
        
        layout.addSpacing(15)
        
        instructions = QLabel(
            "• UUD (proton) quarks shown\n"
            "• UDD (neutron) quarks shown\n"
            "• Gluon flux tubes visible\n"
            "• Nuclear binding forces\n"
            "• Orbitals breathe (quantum)\n"
            "• Zoom in to see labels\n"
            "• Drag to rotate view\n"
            "• Scroll to zoom"
        )
        instructions.setStyleSheet("""
            background-color: rgba(102, 126, 234, 0.1);
            color: #969696;
            padding: 12px;
            border-radius: 5px;
            font-size: 11px;
        """)
        layout.addWidget(instructions)
        
        layout.addStretch()
        
        return sidebar
    
    def create_right_sidebar(self):
        sidebar = QWidget()
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(220)
        sidebar.setStyleSheet("background-color: #0f0f19;")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        orbital_group = QGroupBox("ELECTRON ORBITALS")
        orbital_group.setStyleSheet("QGroupBox { color: #667eea; font-weight: bold; font-size: 13px; padding-top: 8px; }")
        orbital_layout = QVBoxLayout()
        orbital_layout.setContentsMargins(5, 10, 5, 5)
        orbital_layout.setSpacing(0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: #0d0d19;
                border: 1px solid #323246;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                background: #0d0d19;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #667eea;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #7c94f5;
            }
        """)
        
        self.checkbox_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_widget)
        self.checkbox_layout.setSpacing(4)
        self.checkbox_layout.setContentsMargins(5, 5, 5, 5)
        scroll.setWidget(self.checkbox_widget)
        
        orbital_layout.addWidget(scroll)
        orbital_group.setLayout(orbital_layout)
        layout.addWidget(orbital_group, stretch=1)
        
        return sidebar
    
    def create_slider(self, label, min_val, max_val, initial, layout, scale=1):
        label_widget = QLabel(label)
        label_widget.setStyleSheet("color: #667eea;")
        layout.addWidget(label_widget)
        
        slider_layout = QHBoxLayout()
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(initial)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #323246;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #667eea;
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -6px 0;
            }
        """)
        
        value_label = QLabel(str(initial if scale == 1 else initial / scale))
        value_label.setStyleSheet("color: white;")
        value_label.setMinimumWidth(50)
        
        slider.valueChanged.connect(lambda v: self.slider_changed(slider, value_label, scale))
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        layout.addLayout(slider_layout)
        
        return slider
    
    def update_loading_spinner(self):
        spinner_chars = ['◐', '◓', '◑', '◒']
        char = spinner_chars[self.loading_spinner_angle % 4]
        self.loading_spinner_angle += 1
        
        current_text = self.loading_label.text()
        if current_text and not current_text.startswith('✓'):
            parts = current_text.split(' ', 1)
            if len(parts) > 1:
                self.loading_label.setText(f"{char} {parts[1]}")
            else:
                self.loading_label.setText(f"{char} Loading...")
    
    def slider_changed(self, slider, label, scale):
        value = slider.value() / scale
        if scale == 1:
            label.setText(str(int(value)))
        else:
            label.setText(f"{value:.2f}")
        
        if slider == self.threshold_slider or slider == self.resolution_slider:
            self.clear_mesh_cache()
            self.schedule_update()
    
    def schedule_update(self):
        self.update_timer.stop()
        self.update_timer.start(300)
    
    def on_z_changed(self):
        self.clear_mesh_cache()
        self.auto_set_neutrons()
        self.update_nucleus_info()
        self.update_electron_config()
    
    def on_neutron_changed(self):
        self.N = self.neutron_spinbox.value()
        self.A = self.Z + self.N
        self.update_nucleus_info()
        self.draw_nucleus()
    
    def auto_set_neutrons(self):
        Z = self.Z_slider.value()
        
        if Z <= 20:
            N = Z
        else:
            N = int(Z * (1.0 + 0.015 * (Z - 20)))
        
        self.neutron_spinbox.setValue(N)
    
    def toggle_nucleus(self, state):
        self.show_nucleus = (state == Qt.Checked)
        self.draw_nucleus()
    
    def toggle_quarks(self, state):
        self.show_quarks = (state == Qt.Checked)
        self.draw_nucleus()
    
    def update_nucleus_info(self):
        self.Z = self.Z_slider.value()
        self.A = self.Z + self.N
        
        if self.A > 0:
            B = NuclearPhysics.binding_energy(self.A, self.Z)
            B_per_A = NuclearPhysics.binding_energy_per_nucleon(self.A, self.Z)
            radius_fm = NuclearPhysics.nuclear_radius(self.A)
            
            info_text = (
                f"Nucleus: {self.Z}p + {self.N}n = {self.A}<br>"
                f"Binding Energy: {B:.1f} MeV<br>"
                f"B/A: {B_per_A:.2f} MeV/nucleon<br>"
                f"Radius: {radius_fm:.2f} fm"
            )
            self.nuclear_label.setText(info_text)
        else:
            self.nuclear_label.setText("No nucleus")
    
    def update_electron_config(self):
        self.Z = self.Z_slider.value()
        self.electron_config = self.get_electron_configuration(self.Z)
        self.electron_config = self.apply_electron_configuration_exceptions(self.electron_config, self.Z)
        
        elements = [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
            "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl", "Pb", "Bi", "Po", "At", "Rn"
        ]
        element = elements[self.Z - 1] if self.Z <= 86 else f"Z={self.Z}"
        
        config_str = {}
        for n, l, count in self.electron_config:
            orbital_name = f"{n}{['s','p','d','f','g','h'][l]}"
            config_str[orbital_name] = count

        config_text = " ".join([f"{orb}^{count}" if count > 1 else orb for orb, count in config_str.items()])
        
        self.title_label.setText(f"{element} Atom")
        self.config_label.setText(config_text)
        
        self.update_orbital_checkboxes()
        self.schedule_update()
    
    def update_orbital_checkboxes(self):
        for i in reversed(range(self.checkbox_layout.count())):
            widget = self.checkbox_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        
        self.orbital_checkboxes = {}
        self.visible_orbitals = set()
        
        color_idx = 0
        for n, l, count in self.electron_config:
            key = (n, l)
            orbital_name = f"{n}{['s','p','d','f','g','h'][l]}"
            
            checkbox = QCheckBox(f"{orbital_name} [{count}e⁻]")
            checkbox.setChecked(True)
            self.visible_orbitals.add(key)
            
            color = self.color_list[color_idx % len(self.color_list)]
            color_rgb = self.colors[color]
            
            checkbox.setStyleSheet(f"""
                QCheckBox {{
                    color: white;
                    spacing: 8px;
                    padding: 5px;
                }}
                QCheckBox::indicator {{
                    width: 18px;
                    height: 18px;
                    border: 2px solid rgb({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)});
                    border-radius: 4px;
                    background-color: #0d0d19;
                }}
                QCheckBox::indicator:checked {{
                    background-color: rgb({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)});
                }}
            """)
            
            checkbox.stateChanged.connect(lambda state, key=(n,l): self.toggle_orbital(key, state))
            
            self.orbital_checkboxes[key] = (checkbox, color)
            self.checkbox_layout.addWidget(checkbox)
            
            color_idx += 1
        
        self.checkbox_layout.addStretch()
    
    def clear_mesh_cache(self):
        for key, mesh_list in self.mesh_cache.items():
            for mesh in mesh_list:
                try:
                    self.view.removeItem(mesh)
                except (ValueError, RuntimeError):
                    pass  # Already removed
        self.mesh_cache = {}
    
    def generate_single_orbital(self, key):
        n, l = key
        
        if key not in self.orbital_checkboxes:
            return
        
        checkbox, color_name = self.orbital_checkboxes[key]
        orbital_name = f"{n}{['s','p','d','f','g','h'][l]}"
        
        print(f"  Generating {orbital_name}...")
        
        # Pick a representative m value for visualization
        # For l>0, use m=0 for the characteristic shape (dumbbell, cloverleaf, etc.)
        # For l=0 (s orbitals), m must be 0 anyway
        m = 0
        
        volume, max_radius = self.generate_volume_grid(
            n, l, m, self.Z, self.resolution
        )
        
        mesh_list = self.draw_marching_cubes_cached(volume, max_radius, color_name)
        self.mesh_cache[key] = mesh_list
    
    def toggle_orbital(self, key, state):
        if state == Qt.Checked:
            self.visible_orbitals.add(key)
            if key in self.mesh_cache:
                # Re-add meshes to view
                for mesh in self.mesh_cache[key]:
                    try:
                        self.view.addItem(mesh)
                    except (ValueError, RuntimeError):
                        pass  # Handle any add errors
            else:
                self.generate_single_orbital(key)
        else:
            self.visible_orbitals.discard(key)
            if key in self.mesh_cache:
                # Actually remove from view instead of just hiding
                for mesh in self.mesh_cache[key]:
                    try:
                        self.view.removeItem(mesh)
                    except (ValueError, RuntimeError):
                        pass  # Already removed
    
    def update_visualization(self):
        self.threshold = self.threshold_slider.value() / 100.0
        self.resolution = self.resolution_slider.value()
        
        self.draw_nucleus()
        
        orbitals_to_generate = []
        for key in self.visible_orbitals:
            if key not in self.mesh_cache:
                orbitals_to_generate.append(key)
        
        if not orbitals_to_generate:
            print("All visible orbitals already cached")
            return
        
        num_orbitals = len(orbitals_to_generate)
        self.loading_label.setText(f"◐ Generating {num_orbitals} orbital{'s' if num_orbitals != 1 else ''}...")
        self.loading_label.show()
        self.loading_timer.start()
        QApplication.processEvents()
        
        print(f"\n{'='*70}")
        print(f"GENERATING ATOMIC STRUCTURE")
        print(f"Element: Z={self.Z}, Nucleus: {self.Z}p + {self.N}n")
        print(f"Quarks: {'UUD/UUD visible' if self.show_quarks else 'Simple nucleons'}")
        print(f"Orbitals to generate: {len(orbitals_to_generate)}/{len(self.visible_orbitals)}")
        print(f"{'='*70}")
        
        orbital_count = 0
        for key in orbitals_to_generate:
            orbital_count += 1
            n, l = key
            
            if key in self.orbital_checkboxes:
                checkbox, color_name = self.orbital_checkboxes[key]
                orbital_name = f"{n}{['s','p','d','f','g','h'][l]}"
                
                self.loading_label.setText(f"Generating {orbital_name}... ({orbital_count}/{num_orbitals})")
                QApplication.processEvents()
                
                self.generate_single_orbital(key)
        
        self.loading_timer.stop()
        self.loading_label.setText("Complete")
        QApplication.processEvents()
        QTimer.singleShot(2000, self.loading_label.hide)
        
        print(f"{'='*70}\n")
    
    def draw_marching_cubes_cached(self, volume, max_radius, color_name):
        resolution = volume.shape[0]
        base_color = self.colors[color_name]
        
        mesh_list = []
        
        num_levels = 25
        thresholds = np.linspace(self.threshold, 0.95, num_levels)
        
        # Calculate spacing used in volume generation
        spacing = 2 * max_radius / (resolution - 1)
        
        for i, level in enumerate(thresholds):
            try:
                verts, faces, normals, values = measure.marching_cubes(
                    volume, 
                    level=level,
                    spacing=(spacing, spacing, spacing)
                )
                
                # Vertices start at (0,0,0) from marching_cubes, shift to center at origin
                # Since our grid goes from -max_radius to +max_radius
                verts = verts - max_radius
                t = i / (num_levels - 1)
                
                step = max_radius * 2 / resolution
                vertex_colors = np.zeros((len(verts), 4))
                
                for v_idx, vert in enumerate(verts):
                    idx_x = int((vert[0] + max_radius) / step)
                    idx_y = int((vert[1] + max_radius) / step)
                    idx_z = int((vert[2] + max_radius) / step)
                    
                    idx_x = max(0, min(resolution - 1, idx_x))
                    idx_y = max(0, min(resolution - 1, idx_y))
                    idx_z = max(0, min(resolution - 1, idx_z))
                    
                    local_density = volume[idx_x, idx_y, idx_z]
                    
                    density_normalized = (local_density - level) / (1.0 - level + 0.01)
                    density_normalized = np.clip(density_normalized, 0, 1)
                    
                    alpha_from_density = np.power(density_normalized, 0.5)
                    layer_alpha = 0.012 + t * 0.028
                    final_alpha = layer_alpha * (0.3 + 0.7 * alpha_from_density)
                    
                    brightness = 0.5 + t * 0.5
                    saturation = 0.6 + t * 0.4
                    
                    vertex_colors[v_idx] = [
                        base_color[0] * brightness * saturation,
                        base_color[1] * brightness * saturation,
                        base_color[2] * brightness * saturation,
                        final_alpha
                    ]
                
                mesh = gl.GLMeshItem(
                    vertexes=verts,
                    faces=faces,
                    vertexColors=vertex_colors,
                    smooth=True,
                    drawEdges=False,
                    glOptions='additive'
                )
                
                self.view.addItem(mesh)
                mesh_list.append(mesh)
                
            except (ValueError, RuntimeError):
                continue
        
        return mesh_list

if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPLETE ATOMIC VISUALIZER - QUARK STRUCTURE + NUCLEAR BINDING")
    print("="*70)
    print("\nFeatures:")
    print("  • UUD (proton) and UDD (neutron) quark display")
    print("  • Gluon flux tubes with color charge")
    print("  • Nuclear binding forces (Liquid Drop Model)")
    print("  • Distance-based quark labels (zoom to see)")
    print("  • Binding energy calculations")
    print("  • Complete electron configurations")
    print("  • Multi-orbital visualization")
    print("  • Dr. Pedro Portugal")
    print("\n" + "="*70 + "\n")
    
    app = QApplication(sys.argv)
    visualizer = QuantumVisualizer()
    visualizer.show()
    sys.exit(app.exec_())