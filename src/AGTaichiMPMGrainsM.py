import math
import h5py
import numpy as np
import sys
import xmlParserGrains
import taichi as ti
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize



# ti.init(arch=ti.cuda, default_fp=ti.f64)
ti.init(arch=ti.gpu)
n = 2  # Number of features
m = 2  # Number of outputs
ord = 2  # Order of the polynomial

final_theta = ti.Matrix.field(6, 2, dtype=ti.f32, shape=())
final_theta2 = ti.Matrix.field(5, 2, dtype=ti.f32, shape=())
intercept_t = ti.Matrix.field(1, 2, dtype=ti.f32, shape=())
count = ti.field(dtype=ti.i32, shape=())  # 関数呼び出し回数を格納するフィールド

m_vec = ti.Matrix.field(2, 2, dtype=ti.f32, shape=())
m_val = ti.field(dtype=ti.f32, shape=(1, 2))


@ti.data_oriented
class AGTaichiMPMGrains:
    def __init__(self, xmlData):
        # material parameters
        self.dp_a = xmlData.integratorData.dp_a
        self.dp_b = xmlData.integratorData.dp_b
        self.kappa = xmlData.integratorData.bulk_modulus
        self.mu = xmlData.integratorData.shear_modulus

        # flip-pic alpha
        self.alpha = xmlData.integratorData.flip_pic_alpha

        # temporal/spatial resolution
        self.dt = xmlData.integratorData.dt
        self.dx = xmlData.gridData.cell_width
        self.invdx = 1.0 / self.dx

        # near earth gravity
        self.g = ti.Vector(xmlData.nearEarthGravityData.g)

        # iteration count
        self.iteration = ti.field(dtype=int, shape=())
        self.iteration[None] = 0

        # max time
        self.max_time = xmlData.integratorData.max_time

        # configuring grid by using the specified grid center and cell width as is
        # min and max will be recomputed because the specified grid size may not agree with the specified cell width

        # compute grid center and tentative grid width
        grid_center = (xmlData.gridData.max + xmlData.gridData.min) * 0.5
        grid_width = xmlData.gridData.max - xmlData.gridData.min
        self.cell_count = np.ceil(grid_width / self.dx).astype(int)

        # recompute grid width, min and max
        grid_width = self.cell_count.astype(float) * self.dx
        self.grid_min = ti.Vector(grid_center - 0.5 * grid_width)
        self.grid_max = ti.Vector(grid_center + 0.5 * grid_width)

        # allocating fields for grid mass and velocity (momentum)
        self.grid_m = ti.field(dtype=float, shape=self.cell_count)
        self.grid_v = ti.Vector.field(2, dtype=float, shape=self.cell_count)
        self.grid_a = ti.Vector.field(2, dtype=float, shape=self.cell_count)
        # for debug
        self.grid_pos = ti.Vector.field(2, dtype=float, shape=np.prod(self.cell_count))
        self.grid_color = ti.field(ti.i32, shape=np.prod(self.cell_count))

        # particles
        rectangle_width = xmlData.rectangleData.max - xmlData.rectangleData.min
        self.particle_ndcount = np.ceil(rectangle_width * xmlData.rectangleData.cell_samples_per_dim / self.dx).astype(int)
        self.particle_count = np.prod(self.particle_ndcount)
        self.particle_init_min = ti.Vector(xmlData.rectangleData.min)
        self.particle_init_cell_samples_per_dim = xmlData.rectangleData.cell_samples_per_dim
        self.particle_init_vel = ti.Vector(xmlData.rectangleData.vel)

        self.particle_hl = 0.5 * self.dx / xmlData.rectangleData.cell_samples_per_dim
        self.particle_volume = (self.dx / xmlData.rectangleData.cell_samples_per_dim)**2
        self.particle_mass = xmlData.rectangleData.density * self.particle_volume

        self.particle_x = ti.Vector.field(2, dtype=float, shape=self.particle_count)
        self.particle_v = ti.Vector.field(2, dtype=float, shape=self.particle_count)
        self.particle_be = ti.Matrix.field(2, 2, dtype=float, shape=self.particle_count)
        # for debug
        self.particle_color_f = ti.field(ti.f32, shape=self.particle_count)
        self.particle_color = ti.field(ti.i32, shape=self.particle_count)

        # static plane list
        self.num_planes = len(xmlData.staticPlaneList)
        self.static_plane_x = ti.Vector.field(2, dtype=float, shape=self.num_planes)
        self.static_plane_n = ti.Vector.field(2, dtype=float, shape=self.num_planes)
        self.static_plane_type = ti.field(ti.i32, shape=self.num_planes)

        for i in range(self.num_planes):
            self.static_plane_x[i] = xmlData.staticPlaneList[i].x
            self.static_plane_n[i] = xmlData.staticPlaneList[i].n
            if xmlData.staticPlaneList[i].isSticky:
                self.static_plane_type[i] = 1
            else:
                self.static_plane_type[i] = 0

    @ti.kernel
    def initialize(self):
        # clear grid values
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector.zero(float, 2)
            self.grid_a[I] = ti.Vector.zero(float, 2)

        # compute grid point locations (for debug)
        for i in range(self.cell_count[0]*self.cell_count[1]):
            gi = i % self.cell_count[0]
            gj = i // self.cell_count[0]
            I = ti.Vector([gi, gj])
            self.grid_pos[i] = self.grid_min + I.cast(float) * self.dx

        # initialize particles
        for i in range(self.particle_count):
            pi = i % self.particle_ndcount[0]
            pj = i // self.particle_ndcount[0]
            r = ti.Vector([ti.random() - 0.5, ti.random() - 0.5])
            _I = ti.Vector([pi, pj]).cast(float) + r

            self.particle_x[i] = self.particle_init_min + (self.dx / self.particle_init_cell_samples_per_dim) * _I
            self.particle_v[i] = self.particle_init_vel
            self.particle_be[i] = ti.Matrix.identity(float, 2)

    # linear basis functions
    @staticmethod
    @ti.func
    def linearN(x):
        absx = ti.abs(x)
        ret = 1.0 - absx if absx < 1.0 else 0.0
        return ret

    @staticmethod
    @ti.func
    def lineardN(x):
        ret = 0.0
        if ti.abs(x) <= 1.0:
            ret = -1.0 if x >= 0.0 else 1.0
        return ret

    @staticmethod
    @ti.func
    def linearStencil():
        return ti.ndrange(2, 2)

    @ti.func
    def linearBase(self, particle_pos):
        return ((particle_pos - self.grid_min) * self.invdx).cast(int)

    @ti.func
    def linearWeightAndGrad(self, particle_pos, grid_pos):
        delta_over_h = (particle_pos - grid_pos) * self.invdx
        wx = self.linearN(delta_over_h[0])
        wy = self.linearN(delta_over_h[1])
        weight = wx * wy
        weight_grad = ti.Vector([wy * self.lineardN(delta_over_h[0]), wx * self.lineardN(delta_over_h[1])]) * self.invdx
        return weight, weight_grad

    # uGIMP basis functions
    @staticmethod
    @ti.func
    def linearIntegral(xp, hl, xi, w):
        diff = ti.abs(xp - xi)
        ret = 0.0
        if diff >= w + hl:
            ret = 0.0
        elif diff >= w - hl:
            ret = ((w + hl - diff) ** 2) / (2.0 * w)
        elif diff >= hl:
            ret = 2.0 * hl * (1.0 - diff / w)
        else:
            ret = 2.0 * hl - (hl * hl + diff * diff) / w
        return ret

    @staticmethod
    @ti.func
    def linearIntegralGrad(xp, hl, xi, w):
        diff = ti.abs(xp - xi)
        sgn = 1.0 if xp - xi >= 0.0 else -1.0
        ret = 0.0
        if diff >= w + hl:
            ret = 0.0
        elif diff >= w - hl:
            ret = -sgn * (w + hl - diff) / w
        elif diff >= hl:
            ret = -sgn * 2.0 * hl / w
        else:
            ret = 2.0 * (xi - xp) / w
        return ret

    @staticmethod
    @ti.func
    def uGIMPStencil():
        return ti.ndrange(3, 3)

    @ti.func
    def uGIMPBase(self, particle_pos):
        return ((particle_pos - self.particle_hl - self.grid_min) * self.invdx).cast(int)

    @ti.func
    def uGIMPWeightAndGrad(self, particle_pos, grid_pos):
        wx = self.linearIntegral(particle_pos[0], self.particle_hl, grid_pos[0], self.dx)
        wy = self.linearIntegral(particle_pos[1], self.particle_hl, grid_pos[1], self.dx)
        weight = wx * wy / self.particle_volume
        weight_grad = ti.Vector([wy * self.linearIntegralGrad(particle_pos[0], self.particle_hl, grid_pos[0], self.dx), wx * self.linearIntegralGrad(particle_pos[1], self.particle_hl, grid_pos[1], self.dx)]) / self.particle_volume
        return weight, weight_grad



    @ti.kernel
    def step(self):
        self.iteration[None] += 1


        #clear grid data
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector.zero(float, 2)
            self.grid_a[I] = ti.Vector.zero(float, 2)


        # particle status update and p2g
        for p in self.particle_x:
            # base = self.linearBase(self.particle_x[p])
            # stencil = self.linearStencil()
            base = self.uGIMPBase(self.particle_x[p])
            stencil = self.uGIMPStencil()

            # compute particle stress
            J = ti.sqrt(self.particle_be[p].determinant())
            be_bar = self.particle_be[p] / J
            dev_be_bar = be_bar - be_bar.trace() * ti.Matrix.identity(float, 2) * 0.5
            tau = ti.Matrix.zero(float, 2, 2)
            if J < 1.0:
                tau = self.kappa * 0.5 * (J+1.0) * (J-1.0) * ti.Matrix.identity(float, 2) + self.mu * dev_be_bar

            # p2g
            for i, j in ti.static(stencil):
                offset = ti.Vector([i, j])
                # grid point position
                gp = self.grid_min + (base + offset).cast(float) * self.dx

                # compute weight and weight grad
                # weight, weight_grad = self.linearWeightAndGrad(self.particle_x[p], gp)
                weight, weight_grad = self.uGIMPWeightAndGrad(self.particle_x[p], gp)

                # internal force
                f_internal = - self.particle_volume * tau @ weight_grad

                # accumulate grid velocity, acceleration and mass
                self.grid_v[base + offset] += weight * self.particle_mass * self.particle_v[p]
                self.grid_a[base + offset] += f_internal
                self.grid_m[base + offset] += weight * self.particle_mass

        # grid update
        for i, j in self.grid_m:
            if self.grid_m[i, j] > 0:
                self.grid_a[i, j] = self.grid_a[i, j] / self.grid_m[i, j] + self.g
                self.grid_v[i, j] = self.grid_v[i, j] / self.grid_m[i, j] + self.dt * self.grid_a[i, j]

                # boundary conditions
                gp = self.grid_min + self.dx * ti.Vector([i, j])
                for s in ti.static(range(self.num_planes)):
                    d = self.static_plane_n[s].dot(gp - self.static_plane_x[s])

                    if d < 0:
                        if self.static_plane_type[s] == 1: # sticky:
                            self.grid_v[i, j] = ti.Vector.zero(float, 2)
                        else: #sliding
                            vnormal = self.static_plane_n[s].dot(self.grid_v[i, j])
                            if vnormal <= 0.0:
                                self.grid_v[i, j] -= vnormal * self.static_plane_n[s]

        # g2p and update deformation status
        for p in self.particle_x:

            # base = self.linearBase(self.particle_x[p])
            # stencil = self.linearStencil()
            base = self.uGIMPBase(self.particle_x[p])
            stencil = self.uGIMPStencil()

            v_pic = ti.Vector.zero(float, 2)
            grid_a = ti.Vector.zero(float, 2)
            vel_grad = ti.Matrix.zero(float, 2, 2)

        # compute velocity gradient and particle velocity
            for i, j in ti.static(stencil):
                offset = ti.Vector([i, j])
                # grid point position
                gp = self.grid_min + (base + offset).cast(float) * self.dx

                # compute weight and weight grad
                # weight, weight_grad = self.linearWeightAndGrad(self.particle_x[p], gp)
                weight, weight_grad = self.uGIMPWeightAndGrad(self.particle_x[p], gp)

                vel_grad += self.grid_v[base + offset].outer_product(weight_grad)
                v_pic += weight * self.grid_v[base + offset]
                grid_a += weight * self.grid_a[base + offset]

            v_flip = self.particle_v[p] + self.dt * grid_a
            self.particle_v[p] = self.alpha * v_flip + (1.0-self.alpha) * v_pic

            # 差を見る
            be = self.particle_be[p] + self.dt * (vel_grad @ self.particle_be[p] + self.particle_be[p] @ vel_grad.transpose())
            self.particle_be[p] = be

            J = ti.sqrt(be.determinant())

            be_bar = be / J

            dev_be_bar = be_bar - be_bar.trace() * ti.Matrix.identity(float, 2) * 0.5
            Ie_bar = be_bar.trace() * 0.5

            #DPモデルを使用した場合どのくらい補正がかかるかをgapに保存

            s_trial = self.mu * (be_bar - ti.Matrix.identity(float, 2) * Ie_bar)
            s_trial_len = s_trial.norm()
            pressure = -self.kappa * 0.5 * (J + 1.0) * (J - 1.0) / J
            sigma_Y = self.dp_a + self.dp_b * pressure
            phi_trial = s_trial_len - sigma_Y
            gap = 0.0
            if phi_trial > 0.0:
                gap = phi_trial

            # kaiki plastic correction　
            #ここから回帰分析

            J = ti.sqrt(be.determinant())
            be_bar = be / J
            dev_be_bar = be_bar - be_bar.trace() * ti.Matrix.identity(float, 2) * 0.5
            Ie_bar = be_bar.trace() * 0.5
            tau = ti.Matrix.zero(float, 2, 2)
            if J < 1.0:

                tau = self.kappa * 0.5 * (J + 1.0) * (J - 1.0) * ti.Matrix.identity(float, 2) + self.mu * dev_be_bar

            os = tau / J

            # 偏差部分を出してる
            new_pre_stress = os - (0.5 * os.trace() * ti.Matrix.identity(float, 2))

            # ここから補正

            # 固有値分解
            a = new_pre_stress[0, 0]
            b = new_pre_stress[0, 1]
            c = new_pre_stress[1, 0]
            d = new_pre_stress[1, 1]

            # 固有値の計算
            trace = a + d
            det = a * d - b * c
            delta = trace * trace - 4 * det

            lambda1 = (trace + ti.sqrt(delta)) / 2
            lambda2 = (trace - ti.sqrt(delta)) / 2
            m_val[0, 0] = lambda1
            m_val[0, 1] = lambda2

            # 固有ベクトルの計算
            # 固有値 lambda1 に対する固有ベクトル
            if b != 0:
                m_vec[None][0, 0] = -b
                m_vec[None][1, 0] = a - lambda1
            elif c != 0:
                m_vec[None][0, 0] = lambda1 - d
                m_vec[None][1, 0] = c
            else:
                m_vec[None][0, 0] = 1
                m_vec[None][1, 0] = 0

            # 固有値 lambda2 に対する固有ベクトル
            if b != 0:
                m_vec[None][0, 1] = -b
                m_vec[None][1, 1] = a - lambda2
            elif c != 0:
                m_vec[None][0, 1] = lambda2 - d
                m_vec[None][1, 1] = c
            else:
                m_vec[None][0, 1] = 0
                m_vec[None][1, 1] = 1

        ######

            ns = ti.Matrix.zero(float, 2, 2)
            D = ti.Matrix.zero(float, 2, 2)
            # 補正スタート

            if m_val[0, 0] >= m_val[0, 1]:

                # 特徴量作成
                x1 = m_val[0, 0]
                x2 = m_val[0, 1]

                x_hat = ti.Matrix([[x1, x2, x1 ** 2, x1 * x2, x2 ** 2]])

                predictions = x_hat @ final_theta2[None] + intercept_t[None]

                D = ti.Matrix([[predictions[0, 0], 0], [0, predictions[0, 1]]])
            else:
                x2 = m_val[0, 0]
                x1 = m_val[0, 1]
                # 特徴量作成

                x_hat = ti.Matrix([[x1, x2, x1 ** 2, x1 * x2, x2 ** 2]])

                predictions = x_hat @ final_theta2[None] + intercept_t[None]

                D = ti.Matrix([[predictions[0, 1], 0], [0, predictions[0, 0]]])


            V = m_vec[None]

            m_vec_inv = V.inverse()
            sn2 = m_vec[None] @ D @  m_vec_inv

            sn2 = sn2 - 0.5 * sn2.trace() * ti.Matrix.identity(float, 2)

            sn = sn2 + 0.5 * os.trace() * ti.Matrix.identity(float, 2)


            for i in ti.static(range(2)):
                for j in ti.static(range(2)):
                    ns[i, j] = sn[i, j]

            # 回帰の補正からさらにgapの分補正する
            if gap != 0:

                ns2 = ns * J
                s = ns2 - self.kappa * 0.5 * (J + 1.0) * (J - 1.0) * ti.Matrix.identity(float, 2)
                s_len = s.norm()
                n = s * (1.0 / s_len)
                s_new = (s_len - gap) * n

                new_be_bar = s_new / self.mu + ti.Matrix.identity(float, 2) * Ie_bar
                be = new_be_bar * J / ti.sqrt(new_be_bar.determinant())


            self.particle_be[p] = be

            # boundary conditions
            for s in ti.static(range(self.num_planes)):
                d = self.static_plane_n[s].dot(self.particle_x[p] - self.static_plane_x[s])

                if d < 0:
                    if self.static_plane_type[s] == 1: # sticky:
                        self.particle_v[p] = ti.Vector.zero(float, 2)
                    else: #sliding
                        vnormal = self.static_plane_n[s].dot(self.particle_v[p])
                        if vnormal <= 0.0:
                            self.particle_v[p] -= vnormal * self.static_plane_n[s]

                # advect
            self.particle_x[p] += self.dt * self.particle_v[p]

    def saveState(self, file_name):
            print('[AGTaichiMPMGrains] saving state to ' + file_name)
            with h5py.File(file_name, 'w') as h5:
                h5.create_group('mpm2d')
                h5['mpm2d'].create_dataset('timestep', data=self.dt)
                h5['mpm2d'].create_dataset('iteration', data=self.iteration[None])
                h5['mpm2d'].create_dataset('time', data=float(self.iteration[None]) * self.dt)
                h5['mpm2d'].create_dataset('dx', data=self.dx)
                h5['mpm2d'].create_dataset('hl', data=self.particle_hl)
                h5['mpm2d'].create_dataset('m', data=self.particle_mass)
                h5['mpm2d'].create_dataset('Vp', data=self.particle_volume)
                h5['mpm2d'].create_dataset('q', data=self.particle_x.to_numpy())
                h5['mpm2d'].create_dataset('v', data=self.particle_v.to_numpy())
                h5['mpm2d'].create_dataset('be', data=self.particle_be.to_numpy())


def f(Theta, X):
        Theta = Theta.reshape(m, X.shape[1])
        return X @ Theta.T

def objective_function(Theta, X, Y):
    n_features = X.shape[1]
    weights = Theta[:n_features * 2].reshape(-1, 2)  # 係数
    intercept = Theta[n_features * 2:]  # 切片

    Y_pred = X @ weights + intercept # 予測値
    mse = np.mean((Y - Y_pred) ** 2)  # 平均二乗誤差
    return mse

def compute_gradient(Theta, X, Y):
        Theta = Theta.reshape(m, X.shape[1])
        predictions =f(Theta, X)
        error = predictions - Y
        gradient = (2 / X.shape[0]) * X.T @ error
        return gradient.T.ravel()



#####################
def get_data2(n, m):
    data = pd.read_csv('p_m_c21_new2.csv')
    # data = pd.read_csv('normal_xdiff_data.csv')
    y = data.iloc[:, m:m + n].values
    x = data.iloc[:, :m].values

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(x)

    print("X_poly shape: ", X_poly.shape[0])
    return X_poly, y


if len(sys.argv) <= 1:
    print("usage: python AGTaichiMPMGrains.py <xml file>")
    exit(-1)

xmlData = xmlParserGrains.MPMGrainsXMLData(sys.argv[1])
xmlData.show()

agTaichiMPMGrains = AGTaichiMPMGrains(xmlData)
agTaichiMPMGrains.initialize()


X_hat2, y2 = get_data2(n, m)
print("X_hat shape: ", X_hat2.shape)
print("y shape: ", y2.shape)

n_features = X_hat2.shape[1]
# 初期パラメータ（固定）

Theta_initial2 = np.zeros(X_hat2.shape[1] * 2 + 2)
# Use the Jacobian in the optimization

result2 = minimize(objective_function, Theta_initial2, args=(X_hat2, y2), method='BFGS')

final_theta_np2 = result2.x.reshape(2, 6)
# Output optimized parameters and cost
optimal_params = result2.x
weights = optimal_params[:X_hat2.shape[1] * 2].reshape(-1, 2)
intercept = optimal_params[n_features * 2:]              # 切片


print("最適な切片:\n", intercept)
print("最適な係数:\n", weights)
for j in ti.static(range(2)):
    intercept_t[None][0, j] = intercept[j]

for i in ti.static(range(5)):
    for k in ti.static(range(2)):
        final_theta2[None][i, k] = weights[i, k]


print("Minimum cost:", result2.fun)

gui = ti.GUI("AGTaichiMPMGrains")
t_count = 0

while True:
    if gui.get_event():
        if gui.event.key == gui.SPACE and gui.event.type == gui.PRESS:
            agTaichiMPMGrains.initialize()
    count[None] = 0
    for i in range(1000):


        agTaichiMPMGrains.step()


        time = agTaichiMPMGrains.iteration[None] * agTaichiMPMGrains.dt

        if time >= agTaichiMPMGrains.max_time:
            agTaichiMPMGrains.saveState('result.h5')
            print('[AGTaichiMPMGrains2D] simulation done.')
            # スクリーンショットを保存
            screenshot_filename = "screenshot5.png"
            gui.circles(agTaichiMPMGrains.particle_x.to_numpy(), radius=2, color=0xFFFFFF)
            gui.show(screenshot_filename)
            print("Screenshot saved as 'screenshot.png'")
            exit(0)
    t_count += 1
    print("all", agTaichiMPMGrains.particle_count)
    print("DP", count[None])
    print(time)
    if (t_count%10) == 0:
        #
        screenshot_filename = "KData_miss_"+str(t_count)+".png"
        gui.circles(agTaichiMPMGrains.particle_x.to_numpy(), radius=2, color=0xFFFFFF)
        gui.show(screenshot_filename)
    gui.circles(agTaichiMPMGrains.particle_x.to_numpy(), radius=2, color=0xFFFFFF)
    gui.show()
