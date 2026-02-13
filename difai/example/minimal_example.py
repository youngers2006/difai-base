from difai.aif import AIF_Env, AIF_Agent, AIF_Simulation
from jax import numpy as jnp
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt


class Minimal_Env(AIF_Env):
    def __init__(self, x0=jnp.array([0.0, 0.0]), dt=0.02, k=1.0):
        self.x0 = x0
        self.dt = dt
        self.sys_params = {'k': k}
        self.dim_sys_params = len(self.sys_params)
        self.non_negative_sys_params = [0]
        self.dim_action = 1
        self.dim_observation = 1
        self.jitable = True

    @staticmethod
    def _forward_complete(x, u, dt, random_realisation, key, k):
        """
        Computes one forward step 
        :param x: initial state (position, velocity)
        :param u: control value (velocity)
        :param dt: time step duration in seconds
        :param k: stiffness parameter
        :return:
            - x: resulting state
        """
        # 1. System matrix f'(x) = A @ x 
        A = jnp.array([[0,   1,  0],    
                       [-k,  0,  1],
                       [0,   0,  0]])  

        step_fn = lambda y, t: A @ y
        
        # 2. (Online) Control Algorithm 
        y = jnp.hstack([x, u]) 
        solution = odeint(step_fn, y, jnp.array([0,dt]), rtol=1.4e-8, atol=1.4e-8)
        y = solution[1]

        x = x.at[:2].set(y[:2])  # update position and velocity
       
        return x 

    @staticmethod
    def _get_observation_complete(x, k):
        return x[:1] # observe position only
    
if __name__ == "__main__":
    ### Inference only example
    x0 = jnp.array([0.0, 0.0])  # Initial state (position, velocity)
    k = 1.0  # Stiffness parameter
    dt = 0.02  # Time step duration in seconds
    generative_process = Minimal_Env(x0=x0, dt=dt, k=k) # Real world
    generative_model = Minimal_Env(x0=x0, dt=dt, k=k) # Agent's internal model (Use the same model for simplicity)

    # Create noise parameters
    noise_params = {'observation_std': {'id': jnp.array([0, 1])}}  # Gaussian noise on observation


    # Create an AIF agent with the generative model and set inital belief and action limits
    agent = AIF_Agent(generative_model, noise_params)
    agent.set_initial_beliefs(initial_belief_state=[x0+0.1, jnp.diag(jnp.array([0.1,0.1])**2)],
                              initial_belief_noise=[jnp.log(jnp.array([0.05, 0.05])), jnp.diag(jnp.array([1.0,1.0]))],
                              initial_belief_sys= [jnp.array([k]), jnp.diag(jnp.array([1]))])
    agent.set_params_with_defaults(a_lims= jnp.array([[-10.0], [10.0]]))  # Action limits

    noise_params = {'observation_std':{'id':jnp.array([0]), 'value': jnp.array([0.05])}}  # Gaussian noise on observation

    # Create a simulation environment with the agent and generative process
    sim = AIF_Simulation(agent=agent, generative_process=generative_process, noise_params=noise_params)

    print("Running inference with minimal environment...")
    bb, xx, oo, aa, aa_applied, lll = sim.run_inference_only(numsteps=100, random_a=True)
    print("Inference completed.")

    # Plotting the results
    t = jnp.arange(0, len(xx) * dt, dt)
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    ax[0].plot(t, [x[0] for x in xx], label='Position (True)', color='blue' )
    ax[0].plot(t[1:], [o[0] for o in oo], label='Position (Observation)', color='orange', linestyle='--')
    mean = jnp.array([b[0][0] for b in bb])
    ax[0].plot(t, mean, label='Position (Belief)', color='purple')
    ribbon = jnp.array([jnp.sqrt(b[1][0,0]) for b in bb])
    ax[0].fill_between(t, mean - ribbon, mean + ribbon, color='purple', alpha=0.2, label='Standard Deviation')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position')
    ax[0].set_title('Position Over Time')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, [x[1] for x in xx], label='Velocity (True)', color='blue')
    mean_vel = jnp.array([b[0][1] for b in bb])
    ax[1].plot(t, mean_vel, label='Velocity (Belief)', color='purple')
    ribbon_vel = jnp.array([jnp.sqrt(b[1][1,1]) for b in bb])
    ax[1].fill_between(t, mean_vel - ribbon_vel, mean_vel + ribbon_vel, color='purple', alpha=0.2, label='Standard Deviation')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Velocity')
    ax[1].set_title('Velocity Over Time')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t[1:], aa, label='Action', color='green')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Action')
    ax[2].set_title('Action Over Time')
    ax[2].legend()
    ax[2].grid()

    fig.savefig('minimal_example_inference.png')
    fig.tight_layout()
    print("Plot saved as 'minimal_example_inference.png'.")

    ### Control to target example
    target = 0.5 # Define a target position
    C=[jnp.array([target]), jnp.array([0.01**2])]  # Preference distribution preferring observations close to the target with a small variance
    agent.set_preference_distribution(C=C, C_index=[0], use_observation_preference=True)  # Set the preference distribution for the agent

    print(f"Run control to target {target}")
    bb, bb_after_rt, xx, oo, aa, aa_applied, lll, NEFE_PLAN, PRAGMATIC_PLAN, INFO_GAIN_PLAN, NEFES, PRAGMATICS, INFO_GAINS = sim.run_aif_perceptual_delay(numsteps=100)
    print("Control completed.")

    # Plotting the results
    t = jnp.arange(0, len(xx) * dt, dt)
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    ax[0].plot(t, target*jnp.ones_like(t), label='Target', color='red' , linestyle='--') # Target line
    ax[0].plot(t, [x[0] for x in xx], label='Position (True)', color='blue' )
    ax[0].plot(t[1:], [o[0] for o in oo], label='Position (Observation)', color='orange', linestyle='--')
    mean = jnp.array([b[0][0] for b in bb])
    ax[0].plot(t, mean, label='Position (Belief)', color='purple')
    ribbon = jnp.array([jnp.sqrt(b[1][0,0]) for b in bb])
    ax[0].fill_between(t, mean - ribbon, mean + ribbon, color='purple', alpha=0.2, label='Standard Deviation')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position')
    ax[0].set_title('Position Over Time')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, [x[1] for x in xx], label='Velocity (True)', color='blue')
    mean_vel = jnp.array([b[0][1] for b in bb])
    ax[1].plot(t, mean_vel, label='Velocity (Belief)', color='purple')
    ribbon_vel = jnp.array([jnp.sqrt(b[1][1,1]) for b in bb])
    ax[1].fill_between(t, mean_vel - ribbon_vel, mean_vel + ribbon_vel, color='purple', alpha=0.2, label='Standard Deviation')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Velocity')
    ax[1].set_title('Velocity Over Time')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t[1:], aa, label='Action', color='green')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Action')
    ax[2].set_title('Action Over Time')
    ax[2].legend()
    ax[2].grid()

    fig.savefig('minimal_example_control.png')
    fig.tight_layout()
    print("Plot saved as 'minimal_example_control.png'.")


