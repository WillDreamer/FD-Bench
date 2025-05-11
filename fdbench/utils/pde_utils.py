import numpy as np
import torch


# def reaction_1(u1, u2):
#     k = 5e-3

#     return u1 - (u1 * u1 * u1) - k - u2
# def reaction_2(u1, u2):
#     return u1 - u2

# def pde_DR(y,x):
#     d1 = 1e-3
#     d2 = 5e-3

#     du1_xx = dde.grad.hessian(y, x, i=0, j=0, component=0)
#     du1_yy = dde.grad.hessian(y, x, i=1, j=1, component=0)
#     du2_xx = dde.grad.hessian(y, x, i=0, j=0, component=1)
#     du2_yy = dde.grad.hessian(y, x, i=1, j=1, component=1)

#     # TODO: check indices of jacobian
#     du1_t = dde.grad.jacobian(y, x, i=0, j=2)
#     du2_t = dde.grad.jacobian(y, x, i=1, j=2)

#     u1 = y[:,0].unsqueeze(1)
#     u2 = y[:,1].unsqueeze(1)

#     eq1 = du1_t - reaction_1(u1, u2) - d1 * (du1_xx + du1_yy)
#     eq2 = du2_t - reaction_2(u1, u2) - d2 * (du2_xx + du2_yy)

#     return eq1 + eq2

def pde_DR(outs, grid, d1=1e-3, d2=5e-3, k=5e-3):
    """
    Reaction-Diffusion system PDE residuals
    u1_t = d1 ∇² u1 + reaction_1(u1, u2)
    u2_t = d2 ∇² u2 + reaction_2(u1, u2)

    Parameters:
        outs: Tensor, [u1, u2] ∈ ℝ^{N×2}
        grid: Tensor, [x, y, t] ∈ ℝ^{N×3}
    """

    u1 = outs[:, 0:1]
    u2 = outs[:, 1:2]

    def grad(f, wrt):
        return torch.autograd.grad(f.sum(), wrt, create_graph=True, retain_graph=True)[0]

    # First derivatives
    u1_grad = grad(u1, grid)
    u2_grad = grad(u2, grid)

    u1_t = u1_grad[:, 2:3]
    u2_t = u2_grad[:, 2:3]

    # Spatial gradients
    u1_x = u1_grad[:, 0:1]
    u1_y = u1_grad[:, 1:2]
    u2_x = u2_grad[:, 0:1]
    u2_y = u2_grad[:, 1:2]

    # Second derivatives (Laplacian)
    u1_xx = grad(u1_x, grid)[:, 0:1]
    u1_yy = grad(u1_y, grid)[:, 1:2]
    u2_xx = grad(u2_x, grid)[:, 0:1]
    u2_yy = grad(u2_y, grid)[:, 1:2]

    lap_u1 = u1_xx + u1_yy
    lap_u2 = u2_xx + u2_yy

    # Reaction terms
    reaction1 = u1 - u1**3 - k - u2
    reaction2 = u1 - u2

    # PDE residuals
    eq1 = u1_t - d1 * lap_u1 - reaction1
    eq2 = u2_t - d2 * lap_u2 - reaction2

    # Optional: return residuals separately
    return eq1 + eq2


# def pde_KF(u, x, resolution=128):
#     X, Y = torch.meshgrid(
#             torch.linspace(0, 1, resolution),
#             torch.linspace(0, 1, resolution),
#             indexing="ij",
#         )
#     f = lambda x, y: 0.1 * torch.sin(2.0 * np.pi * (x + y))
#     forcing = f(X, Y).unsqueeze(0)
    
#     u_vel, v_vel = u[:,0], u[:,1]

#     u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
#     u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
#     u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

#     v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
#     v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
#     v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

#     w_vor_x = dde.grad.jacobian(u, x, i=2, j=0)
#     w_vor_y = dde.grad.jacobian(u, x, i=2, j=1)
#     w_vor_t = dde.grad.jacobian(u, x, i=2, j=2)

#     w_vor_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
#     w_vor_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)

#     eqn1 = w_vor_t + u_vel * w_vor_x + v_vel * w_vor_y - \
#            0.001 * (w_vor_xx + w_vor_yy) - forcing(x)
#     eqn2 = u_vel_x + v_vel_y
#     eqn3 = u_vel_xx + u_vel_yy + w_vor_y
#     eqn4 = v_vel_xx + v_vel_yy - w_vor_x
#     return eqn1 + eqn2 + eqn3 + eqn4

def pde_KF(outs, grid, resolution=128, nu=0.001, device='cuda'):
    """
    Kelvin-Helmholtz (KF) PDE residuals with autograd.

    outs: [u, v, w] shape [N, 3]
    grid: [x, y, t] shape [N, 3]
    """

    u_vel = outs[:, 0:1]  # u
    v_vel = outs[:, 1:2]  # v
    w_vor = outs[:, 2:3]  # vorticity ω

    def grad(f, wrt):
        return torch.autograd.grad(f.sum(), wrt, create_graph=True, retain_graph=True)[0]

    # First-order gradients
    u_grad = grad(u_vel, grid)  # shape: [N, 3]
    v_grad = grad(v_vel, grid)
    w_grad = grad(w_vor, grid)

    u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]
    v_y = v_grad[:, 1:2]

    w_x = w_grad[:, 0:1]
    w_y = w_grad[:, 1:2]
    w_t = w_grad[:, 2:3]

    # Second-order gradients
    w_xx = grad(w_x, grid)[:, 0:1]
    w_yy = grad(w_y, grid)[:, 1:2]
    u_xx = grad(u_x, grid)[:, 0:1]
    u_yy = grad(u_y, grid)[:, 1:2]
    v_xx = grad(v_grad[:, 0:1], grid)[:, 0:1]
    v_yy = grad(v_y, grid)[:, 1:2]

    # Forcing term
    # Generate static forcing field at runtime on correct device
    X, Y = torch.meshgrid(
        torch.linspace(0, 1, resolution, device=device),
        torch.linspace(0, 1, resolution, device=device),
        indexing="ij"
    )
    f = lambda x, y: 0.1 * torch.sin(2.0 * np.pi * (x + y))
    forcing_map = f(X, Y).reshape(1, 1, resolution, resolution)  # [1, 1, H, W]

    # Map from grid [N, 3] to forcing value
    # Normalize x,y to [0, resolution-1]
    x_norm = (grid[:, 0] * (resolution - 1)).long().clamp(0, resolution - 1)
    y_norm = (grid[:, 1] * (resolution - 1)).long().clamp(0, resolution - 1)
    forcing_val = forcing_map[0, 0, x_norm, y_norm].unsqueeze(1)

    # Residuals
    eqn1 = w_t + u_vel * w_x + v_vel * w_y - nu * (w_xx + w_yy) - forcing_val
    eqn2 = u_x + v_y
    eqn3 = u_xx + u_yy + w_y
    eqn4 = v_xx + v_yy - w_x

    # Return each term if needed separately, or sum
    return eqn1 + eqn2 + eqn3 + eqn4

def pde_CNS(outs, grid, gamma=1.6):

    rho      = outs[:, 0:1]
    vel_u    = outs[:, 1:2]
    vel_v    = outs[:, 2:3]
    pressure = outs[:, 3:4]

    E = pressure / (gamma - 1.0) + 0.5 * rho * (vel_u**2 + vel_v**2)

    def grad(f, wrt):
        return torch.autograd.grad(
            f.sum(), wrt,
            create_graph=True, retain_graph=True
        )[0]

    u_grad = grad(vel_u, grid)
    v_grad = grad(vel_v, grid)
    p_grad = grad(pressure, grid)

    u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]
    v_x, v_y = v_grad[:, 0:1], v_grad[:, 1:2]
    p_x, p_y = p_grad[:, 0:1], p_grad[:, 1:2]

    u_xx = grad(u_x, grid)[:, 0:1]
    u_yy = grad(u_y, grid)[:, 1:2]
    v_xx = grad(v_x, grid)[:, 0:1]
    v_yy = grad(v_y, grid)[:, 1:2]

    rho_u_grad = grad(rho * vel_u, grid)
    rho_v_grad = grad(rho * vel_v, grid)
    continuity = rho_u_grad[:, 0:1] + rho_v_grad[:, 1:2]

    rho_uu_grad = grad(rho * vel_u * vel_u, grid)
    rho_uv_grad = grad(rho * vel_u * vel_v, grid)
    momentum_x = rho_uu_grad[:, 0:1] + rho_uv_grad[:, 1:2] + p_x - (u_xx + u_yy)

    rho_vu_grad = grad(rho * vel_v * vel_u, grid)
    rho_vv_grad = grad(rho * vel_v * vel_v, grid)
    momentum_y = rho_vu_grad[:, 0:1] + rho_vv_grad[:, 1:2] + p_y - (v_xx + v_yy)

    flux_E_x = grad((E + pressure) * vel_u, grid)[:, 0:1]
    flux_E_y = grad((E + pressure) * vel_v, grid)[:, 1:2]
    energy = flux_E_x + flux_E_y

    mse = torch.nn.MSELoss()
    zeros = lambda var: torch.zeros_like(var, dtype=torch.float32, device=var.device)
    loss = (
        mse(continuity, zeros(continuity)) +
        mse(momentum_x, zeros(momentum_x)) +
        mse(momentum_y, zeros(momentum_y)) +
        mse(energy, zeros(energy))
    )

    return loss

