import numpy as np

def non_markovian_midprice(inital_prices, 
                        permenant_price_impact_func, 
                        nu_vals,
                        kernel_function, 
                        dt, 
                        rng,
                        num_paths,
                        num_timesteps,
                        sigma):
    
    noise_process = rng.normal(0,1, size=(num_timesteps, num_paths)) 
    brownian_motion_integrand = sigma*(dt**0.5)*noise_process
    zeros = np.zeros((1, num_paths))
    brownian_motion_integrand = np.vstack([zeros, brownian_motion_integrand])
    brownian_motion_integral = np.cumsum(brownian_motion_integrand, axis=0)

    prices = []
    for super_timestep in range(0, num_timesteps):
        final_time = super_timestep*dt

        permenant_price_impact_vals = permenant_price_impact_func(nu_vals[:super_timestep])
        # This gives K(s, t) for s<=t to integrate over
        kernel_vals = np.array([kernel_function(timestep*dt, final_time) for timestep in range(super_timestep)]) 
        kernel_integrand = kernel_vals.reshape(-1, 1) * permenant_price_impact_vals * dt

        # We do a vstack as the axis=1 are the batches
        kernel_integrand = np.vstack([inital_prices, kernel_integrand])
        kernel_integral = np.sum(kernel_integrand, axis=0)

        s_t = kernel_integral + brownian_motion_integral[super_timestep]

        prices.append(s_t)
    
    prices = np.vstack(prices)
    
    return prices, noise_process

# def non_markovian_midprice(inital_prices, 
#                         permenant_price_impact_func, 
#                         nu_vals,
#                         kernel_function, 
#                         dt, 
#                         rng,
#                         num_paths,
#                         num_timesteps,
#                         sigma):

#     final_time = num_timesteps*dt

#     permenant_price_impact_vals = permenant_price_impact_func(nu_vals)

#     # This gives K(s, t) for s<=t to integrate over
#     kernel_vals = np.array([kernel_function(timestep*dt, final_time) for timestep in range(num_timesteps)])
#     kernel_integrand = kernel_vals.reshape(-1, 1) * permenant_price_impact_vals * dt

#     # We do a vstack as the axis=1 are the batches
#     kernel_integrand = np.vstack([inital_prices, kernel_integrand])
#     kernel_integral = np.cumsum(kernel_integrand, axis=0)

#     noise_process = rng.normal(0,1, size=(num_timesteps, num_paths)) 
#     brownian_motion_integrand = sigma*(dt**0.5)*noise_process
#     zeros = np.zeros((1, num_paths))
#     brownian_motion_integrand = np.vstack([zeros, brownian_motion_integrand])

#     brownian_motion_integral = np.cumsum(brownian_motion_integrand, axis=0)
#     prices = kernel_integral+brownian_motion_integral

#     return prices, noise_process