# 🤖 Double Pendulum Simulation Using PINN

A machine-learning model, built with Physics-Informed Neural Networks (PINN), to predict the chaotic motion of a double pendulum.

## ✨ Features

* **Physics-based Data Generation:** A classical solver generates ground-truth time-series trajectories for various initial conditions.
* **PINN Model:** The neural network incorporates physics constraints directly into the loss function, using the system's governing differential equations.
* **Chaotic Motion Prediction:** The trained model aims to accurately predict the highly non-linear and chaotic trajectories
* **Visual Analytic Tools:** Plots and animations to compare model predictions against the ground-truth trajectories for in-depth evaluation.
* **Configurable Parameters:** Allows experimentation with initial conditions and physical properties to test the model's robustness.

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zanko-mezdrichki/Double-Pendulum-Simulation-Using-PINN.git](https://github.com/zanko-mezdrichki/Double-Pendulum-Simulation-Using-PINN.git)
    cd Double-Pendulum-Simulation-Using-PINN
    ```

2.  **Install dependencies:**
    ```bash
    pip install simpy numpy matplotlib pytorch scipy
    ```
    *(Note: A GPU is highly recommended for faster training)*

3.  **Start the program:**
    ```bash
    python main.py
    ```
4.  Enter the configuration values when prompted.

## ⚙️ Parameters & Configuration

On startup, `main.py` will prompt you to configure the simulation and model. This is key for experimenting with the system's chaotic nature.

- **Initial Conditions:** $\theta_1$, $\dot{\theta}_1$, $\theta_2$, $\dot{\theta}_2$
- **Physical Properties:** $m_1$, $m_2$, $L_1$, $L_2$, $g$
- **Model Hyperparameters:** Time of the simmulation, number of time points considered, learning rate, epochs.

## 🕹️ Preview

For more details and results, see double_pendulum.pdf(highly recommended).

In the preview folder there are images showing the training history and the comparison between angles calculated with the two methods with default parameters.

To see the default animation: [https://youtu.be/9tKttP7lyuA](https://youtu.be/9tKttP7lyuA)

## 📈 Future Improvements

- PyQt5 interface for configuration.
- Implementation of adaptive loss weighting(e.g. GradNorm) for better training balance.
- Testing other architectures such as SIREN or Fourier-feature networks.

## 🛠️ Technologies Used

- Python 3.x
- PyTorch 
- SimPy 
- NumPy
- Matplotlib 
- SciPy

## 📄 License
This project is licensed under the [MIT License](./LICENSE). See the LICENSE file for more details.
