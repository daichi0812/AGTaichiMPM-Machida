# https://docs.python.org/ja/3/library/xml.etree.elementtree.html

import numpy as np
import xml.etree.ElementTree as ET

class MPMGrainsXMLIntegratorData:
    def __init__(self):
        self.dt = 0.0
        self.bulk_modulus = 1.0
        self.shear_modulus = 1.0
        self.dp_a = 0.0
        self.dp_b = 0.0
        self.flip_pic_alpha = 0.95
        self.max_time = 0.0

    def setFromXMLNode(self, node):
        self.dt = float(node.attrib['dt'])
        self.bulk_modulus = float(node.attrib['bulk_modulus'])
        self.shear_modulus = float(node.attrib['shear_modulus'])
        self.dp_a = float(node.attrib['dp_a'])
        self.dp_b = float(node.attrib['dp_b'])
        self.flip_pic_alpha = float(node.attrib['flip_pic_alpha'])
        self.max_time = float(node.attrib['max_time'])

    def show(self):
        print('*** Integrator ***')
        print('  dt: ' + str(self.dt))
        print('  bulk_modulus: ' + str(self.bulk_modulus))
        print('  shear_modulus: ' + str(self.shear_modulus))
        print('  dp_a: ' + str(self.dp_a))
        print('  dp_b: ' + str(self.dp_b))
        print('  flip_pic_alpha: ' + str(self.flip_pic_alpha))
        print('  max_time: ' + str(self.max_time))

class MPMXMLGridData:
    def __init__(self, dim):
        self.min = np.zeros(dim)
        self.max = np.zeros(dim)
        self.cell_width = 1.0

    def setFromXMLNode(self, node):
        self.min = np.array([float(e) for e in node.attrib['min'].split(' ')])
        self.max = np.array([float(e) for e in node.attrib['max'].split(' ')])
        self.cell_width = float(node.attrib['cell_width'])

    def show(self):
        print('*** Grid ***')
        print('  min: ' + str(self.min))
        print('  max: ' + str(self.max))
        print('  cell_width: ' + str(self.cell_width))

class MPMXMLRectangleData:
    def __init__(self, dim):
        self.min = np.zeros(dim)
        self.max = np.zeros(dim)
        self.density = 1.0
        self.cell_samples_per_dim = 2
        self.vel = np.zeros(dim)

    def setFromXMLNode(self, node):
        self.min = np.array([float(e) for e in node.attrib['min'].split(' ')])
        self.max = np.array([float(e) for e in node.attrib['max'].split(' ')])
        self.density = float(node.attrib['density'])
        self.cell_samples_per_dim = int(node.attrib['cell_samples_per_dim'])
        self.vel = np.array([float(e) for e in node.attrib['vel'].split(' ')])

    def show(self):
        print('*** Rectangle ***')
        print('  min: ' + str(self.min))
        print('  max: ' + str(self.max))
        print('  density: ' + str(self.density))
        print('  cell_samples_per_dim: ' + str(self.cell_samples_per_dim))
        print('  vel: ' + str(self.vel))

class MPMXMLStaticPlaneData:
    def __init__(self, dim):
        self.x = np.zeros(dim)
        self.n = np.zeros(dim)
        self.isSticky = False

    def setFromXMLNode(self, node):
        self.x = np.array([float(e) for e in node.attrib['x'].split(' ')])
        self.n = np.array([float(e) for e in node.attrib['n'].split(' ')])
        if node.attrib['boundary_behavior'] == 'sticking':
            self.isSticky = True
        else:
            self.isSticky = False

    def show(self):
        print('*** Static plane ***')
        print('  x: ' + str(self.x))
        print('  n: ' + str(self.n))
        print('  isSticky: ' + str(self.isSticky))

class MPMXMLNearEarthGravityData:
    def __init__(self, dim):
        self.g = np.zeros(dim)

    def setFromXMLNode(self, node):
        self.g = np.array([float(e) for e in node.attrib['f'].split(' ')])

    def show(self):
        print('*** Near earth gravity ***')
        print('  g: ' + str(self.g))

class MPMGrainsXMLData:
    def __init__(self, file_name):
        print('[AGTaichiMPMGrains2D] Parsing xml file: ' + str(file_name))
        tree = ET.parse(file_name)
        root = tree.getroot()
        if root.tag != 'AGTaichiMPMGrains2D':
            print('[AGTaichiMPMGrains2D] Could not find root note AGTaichiMPMGrains2D. Exiting...')
            exit(-1)

        integrator = root.find('integrator')
        self.integratorData = MPMGrainsXMLIntegratorData()
        self.integratorData.setFromXMLNode(integrator)

        grid = root.find('grid')
        self.gridData = MPMXMLGridData(2)
        self.gridData.setFromXMLNode(grid)

        rectangle = root.find('rectangle')
        self.rectangleData = MPMXMLRectangleData(2)
        self.rectangleData.setFromXMLNode(rectangle)

        self.staticPlaneList = []
        for static_plane in root.findall('static_plane'):
            staticPlaneData = MPMXMLStaticPlaneData(2)
            staticPlaneData.setFromXMLNode(static_plane)
            self.staticPlaneList.append(staticPlaneData)

        nearEarthGravity = root.find('near_earth_gravity')
        self.nearEarthGravityData = MPMXMLNearEarthGravityData(2)
        self.nearEarthGravityData.setFromXMLNode(nearEarthGravity)

    def show(self):
        print('[AGTaichiMPMGrains2D] XML Data:')
        self.integratorData.show()
        self.gridData.show()
        self.rectangleData.show()
        self.nearEarthGravityData.show()
        for sp in self.staticPlaneList:
            sp.show()
