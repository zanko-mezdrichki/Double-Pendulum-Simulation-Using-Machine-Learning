# Double Pendulum Simulation Using PINN

A Physics-Informed Neural Networks (**PINN**) created to predict the chaotic motion of a double pendulum. 

The result is compared with solutions calculated using the **odeint** function from SciPy.

# Features

* **Physics-based Data Generation:** SciPy's odeint function generates trajectories for various initial conditions.
* **PINN Model:** PINN incorporates the Euler-Lagrange equations directly into the loss function.
* **Chaotic Motion Prediction:** The trained model aims to accurately predict the highly non-linear and chaotic trajectories.
* **Visual Analytic Tools:** Plots and animations to compare model predictions against the ground-truth trajectories.
* **Configurable Parameters:** Allows experimentation with initial conditions and PINN properties.

# Installation & Usage

*(Note: A GPU is highly recommended for faster training)*

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zanko-mezdrichki/Double-Pendulum-Simulation-Using-PINN.git](https://github.com/zanko-mezdrichki/Double-Pendulum-Simulation-Using-PINN.git)
    cd Double-Pendulum-Simulation-Using-PINN
    ```

2.  **Install dependencies:**
    ```bash
    pip install simpy numpy matplotlib pytorch scipy
    ```

3.  **Start the program:**
    ```bash
    python main.py
    ```
4.  Enter the configuration values when prompted.

# Parameters & Configuration

- **Initial Conditions:** $\theta_1$, $\dot{\theta}_1$, $\theta_2$, $\dot{\theta}_2$
- **Physical Properties:** $m_1$, $m_2$, $L_1$, $L_2$, $g$
- **Model Hyperparameters:** Duration of the simulation, number of time points considered, learning rate, epochs.

# Results

For more details and results, see **double_pendulum.pdf** (highly recommended).

In the preview folder there are images showing the training history and the comparison between angles calculated with the two methods with default parameters.

To see the animation with default parameters: [https://youtu.be/9tKttP7lyuA](https://youtu.be/9tKttP7lyuA)

# Future Improvements

- PyQt5 interface for configuration.
- Implementation of adaptive loss weighting(e.g. **GradNorm**) for better training balance.
- Testing other architectures such as **SIREN** or **Fourier-feature networks**.

# Technologies Used

- Python 3.12.11
- PyTorch - for PINN
- SimPy - for symbolic equations
- NumPy - for data management
- Matplotlib - for charts and animations
- SciPy - for the evaluation

# License
This project is licensed under the [MIT License](./LICENSE). See the LICENSE file for more details.
