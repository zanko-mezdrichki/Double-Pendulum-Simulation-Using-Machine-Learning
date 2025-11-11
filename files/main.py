import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn

from config import obtain_config
from physics import true_simulation
from AI_model import PINN

#in order to obtain similar results(same casual initial points)
seed = 45
torch.manual_seed(seed)

# Configuration and true data
config = obtain_config()
time_list, true_solution = true_simulation(config)

theta1_true = true_solution[:, 0] #first column 
theta2_true = true_solution[:, 2]

L1 = config['L1']
L2 = config['L2']
g_val = config['g']
m1_val = config['m1']
m2_val = config['m2']
L1_val = config['L1']
L2_val = config['L2']

t_max = config['t_max']
n_points = config['n_points']

# Initial condition
y0_tensor = torch.tensor(
    [config['theta1_0'], config['dtheta1_0'], config['theta2_0'], config['dtheta2_0']],
    dtype=torch.float32
).view(1, 4) #1x4 matrix

t_zero = torch.tensor([0.0], dtype=torch.float32, requires_grad=True).view(1, 1)

# Collocation points for physics loss
n_collocation =int(1.5 * n_points)  
t_collocation_np = np.linspace(0, t_max, n_collocation)
t_physics = torch.tensor(t_collocation_np, dtype=torch.float32).view(-1, 1) #-1=n_collocation
t_physics.requires_grad_(True)

# Data points for improve learning
n_data_points = int(n_points/20)
data_indices = np.linspace(0, len(time_list)-1, n_data_points, dtype=int)
t_data = torch.tensor(time_list[data_indices], dtype=torch.float32).view(-1, 1)
values_data = torch.tensor(true_solution[data_indices], dtype=torch.float32)

#setting the model
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']) #Adam optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=75, factor=0.4) #scheduler modify lr in function of the total loss(efficient training)
criterion = nn.MSELoss() #criterion that measures the mean squared error

# Derivative helper
def derivative(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

def initial_condition_loss(model, t_zero_tensor, y0_tensor):
    y0_pred = model(t_zero_tensor)
    return criterion(y0_pred, y0_tensor)

def physics_loss(model, t_physics_tensor, g, m1, m2, L1, L2):
    y_pred = model(t_physics_tensor)
    theta1, dtheta1, theta2, dtheta2 = (
        y_pred[:, 0:1], y_pred[:, 1:2], y_pred[:, 2:3], y_pred[:, 3:4] # to mantain (n_collocation,1) structure
    )
    # 1. Consistency Loss
    theta1_t = derivative(theta1, t_physics_tensor)
    theta2_t = derivative(theta2, t_physics_tensor)
    loss_derivatives = criterion(dtheta1, theta1_t) + criterion(dtheta2, theta2_t) #loss for dtheta_pred and derivative(theta)

    # 2. Eulero_Lagrange solutions
    ddtheta1_autograd = derivative(dtheta1, t_physics_tensor)
    ddtheta2_autograd = derivative(dtheta2, t_physics_tensor)

    c = torch.cos
    s = torch.sin
    
    delta = theta1 - theta2
    den = (m1 + m2 * s(delta)**2)
    
    ddtheta1_formula = (
        m2 * g * s(theta2) * c(delta) - 
        m2 * s(delta) * (L1 * dtheta1**2 * c(delta) + L2 * dtheta2**2) - 
        (m1 + m2) * g * s(theta1)
    ) / (L1 * den)
    
    ddtheta2_formula = (
        (m1 + m2) * (L1 * dtheta1**2 * s(delta) - g * s(theta2) + g * s(theta1) * c(delta)) + 
        m2 * L2 * dtheta2**2 * s(delta) * c(delta)
    ) / (L2 * den)
    
    loss_el = criterion(ddtheta1_autograd, ddtheta1_formula) + \
               criterion(ddtheta2_autograd, ddtheta2_formula) 
    #loss for ddtheta_pred and derivative(dtheta)
    return loss_el + loss_derivatives

# Loss weights
lambda_ic = 800.0   #initial condition for double pendulum is essential 
lambda_phys = 1.0    # physics is chaotic
lambda_data = 70.0   #big importance true data

#training
model.train()
losses_history = {'total': [], 'ic': [], 'data': [], 'physics': []} #for a chart

# epochs
epochs = config['epochs']

for epoch in range(epochs):
    optimizer.zero_grad() #gradient resetting
    
    # Three loss components
    l_ic = initial_condition_loss(model, t_zero, y0_tensor)

    l_phys = physics_loss(model, t_physics, g_val, m1_val, m2_val, L1_val, L2_val)

    values_data_pred = model(t_data)
    l_data = criterion(values_data_pred, values_data)
    
    # Combined loss
    loss = lambda_ic * l_ic + lambda_phys * l_phys + lambda_data * l_data
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step() #adjusting weights to minimize the loss
    scheduler.step(loss.item())  # Adjust learning rate with scheduler
     
    #for graphics
    losses_history['total'].append(loss.item())
    losses_history['ic'].append(l_ic.item())
    losses_history['data'].append(l_data.item())
    losses_history['physics'].append(l_phys.item())
    
    if (epoch + 1) % 100 == 0 or (epoch + 1)==epochs:
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Total Loss = {loss.item():.6f}, "
              f"IC Loss = {l_ic.item():.8f}, "
              f"Data Loss = {l_data.item():.6f}, "
              f"Physics Loss = {l_phys.item():.6f}")

# Training history
plt.figure(figsize=(12, 6))
plt.plot(losses_history['total'], label='Total Loss', linewidth=2)
plt.plot(losses_history['ic'], label='IC Loss', alpha=0.7)
plt.plot(losses_history['data'], label='Data Loss', alpha=0.7)
plt.plot(losses_history['physics'], label='Physics Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_history.png", dpi=300)

# Evaluation
model.eval()
t_eval = torch.tensor(time_list, dtype=torch.float32).view(-1, 1)
with torch.no_grad():
    y_pred = model(t_eval).numpy()

theta1_pred = y_pred[:, 0]
theta2_pred = y_pred[:, 2]

# Calculate errors
theta1_error = np.mean(np.abs(theta1_true - theta1_pred))
theta2_error = np.mean(np.abs(theta2_true - theta2_pred))
theta1_max_error = np.max(np.abs(theta1_true - theta1_pred))
theta2_max_error = np.max(np.abs(theta2_true - theta2_pred))

# Comparison chart angles
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time_list, theta1_true, 'k-', label='SciPy Prediction', linewidth=2)
plt.plot(time_list, theta1_pred, 'r--', label='PINN Prediction', linewidth=2, alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel(r'$\theta_1$ (rad)')
plt.title(f'PINN vs. SciPy Comparison - θ1 (MAE: {theta1_error:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(time_list, theta2_true, 'k-', label='SciPy Prediction', linewidth=2)
plt.plot(time_list, theta2_pred, 'b--', label='PINN Prediction', linewidth=2, alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel(r'$\theta_2$ (rad)')
plt.title(f'PINN vs. SciPy Comparison - θ2 (MAE: {theta2_error:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pinn_comparison.png", dpi=300)

# Animation
#scipy
x1_true = L1 * np.sin(theta1_true)
y1_true = -L1 * np.cos(theta1_true)
x2_true = x1_true + L2 * np.sin(theta2_true)
y2_true = y1_true - L2 * np.cos(theta2_true)

#pinn
x1_pred = L1 * np.sin(theta1_pred)
y1_pred = -L1 * np.cos(theta1_pred)
x2_pred = x1_pred + L2 * np.sin(theta2_pred)
y2_pred = y1_pred - L2 * np.cos(theta2_pred)

#graphs
fig, ax = plt.subplots(1,2, figsize=(16, 8))
max_L = L1 + L2 + 0.5
scipy_ax = ax[0]
scipy_ax.set_xlim(-max_L, max_L)
scipy_ax.set_ylim(-max_L, max_L)
scipy_ax.set_aspect('equal') #same unity
scipy_ax.grid(True)
scipy_ax.set_title("Scipy Simulation")

scipy_line_1, = scipy_ax.plot([], [], 'o-', lw=2, color='blue', label='Scipy Pendulum 1') #lists for lines
scipy_line_2, = scipy_ax.plot([], [], 'o-', lw=2, color='red', label='Scipy Pendulum 2')
scipy_trace, = scipy_ax.plot([], [], ':', lw=1, color='gray', alpha=0.7) #alpha=opacity
scipy_ax.legend(loc='upper right')

ax_pinn=ax[1]
ax_pinn.set_xlim(-max_L, max_L)
ax_pinn.set_ylim(-max_L, max_L)
ax_pinn.set_aspect('equal')
ax_pinn.grid(True)
ax_pinn.set_title("PINN Simulation")

pinn_line_1, = ax_pinn.plot([], [], 'o-', lw=2, color='green', label='PINN Pendulum 1')
pinn_line_2, = ax_pinn.plot([], [], 'o-', lw=2, color='orange', label='PINN Pendulum 2')
pinn_trace, = ax_pinn.plot([], [], ':', lw=1, color='gray', alpha=0.7)
ax_pinn.legend(loc='upper right')

time_text = scipy_ax.text(0.05, 0.9, '', transform=scipy_ax.transAxes)

def update(i):
    scipy_x_p1 = [0, x1_true[i]]
    scipy_y_p1 = [0, y1_true[i]]
    scipy_line_1.set_data(scipy_x_p1, scipy_y_p1) #(0,0)--(x1,y1)
    
    scipy_x_p2 = [x1_true[i], x2_true[i]]
    scipy_y_p2 = [y1_true[i], y2_true[i]]
    scipy_line_2.set_data(scipy_x_p2, scipy_y_p2) #(x1,y1)--(x2,y2)
    
    scipy_trace.set_data(x2_true[:i], y2_true[:i])

    x_p1_pinn = [0, x1_pred[i]]
    y_p1_pinn = [0, y1_pred[i]]
    pinn_line_1.set_data(x_p1_pinn, y_p1_pinn)
    
    x_p2_pinn = [x1_pred[i], x2_pred[i]]
    y_p2_pinn = [y1_pred[i], y2_pred[i]]
    pinn_line_2.set_data(x_p2_pinn, y_p2_pinn)
    
    pinn_trace.set_data(x2_pred[:i], y2_pred[:i])

    time_text.set_text(f'Time: {time_list[i]:.1f}s')
    
    return scipy_line_1, scipy_line_2, scipy_trace, pinn_line_1, pinn_line_2, pinn_trace, time_text

ani_comp = FuncAnimation(
    fig,
    update,
    frames=n_points,
    interval=int(n_points/100),
    blit=True
    )
ani_comp.save('pendulum_simulations.mp4', writer='ffmpeg', fps=100)


print(f"  MAE: θ1={theta1_error:.6f}, θ2={theta2_error:.6f}")
print("See also the animations and the graphs!")


