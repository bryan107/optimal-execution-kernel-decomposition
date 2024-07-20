import numpy as np
import torch

def generate_time_one_time_two(rng,
                                num_timesteps,
                                batch_size,
                                dt,
                                one_direction: bool = False,
                                end_sample_frac: int = 0):
    
    lower_bound = max((num_timesteps-1)*end_sample_frac, 0)
    upper_bound = num_timesteps-1

    if one_direction:
        val_one = rng.uniform(lower_bound+1, upper_bound, batch_size)
        val_two = np.array([rng.uniform(lower_bound, _) for _ in val_one])
    else:
        val_one = rng.uniform(-upper_bound*2, upper_bound*2, batch_size)
        val_two = rng.uniform(-upper_bound*2, upper_bound*2, batch_size)

    time_one = (
                torch.FloatTensor(val_one).reshape(
                    -1, 1
                )
                * dt
            )
    
    time_two = (
        torch.FloatTensor(val_two).reshape(
            -1, 1
        )
        * dt
    )

    return time_one, time_two

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

def vol_non_markovian_midprice(inital_prices, 
                           permenant_price_impact_func, 
                           nu_vals,
                           drift_kernel_function, 
                           vol_kernel_function,
                           dt, 
                           rng,
                           num_paths,
                           num_timesteps,
                           sigma):
    
    noise_process = rng.normal(0,1, size=(num_timesteps, num_paths))

    prices = []
    for super_timestep in range(0, num_timesteps):
        final_time = super_timestep*dt

        permenant_price_impact_vals = permenant_price_impact_func(nu_vals[:super_timestep])

        # This gives K1(s, t) for s<=t to integrate over
        kernel_vals = np.array([drift_kernel_function(timestep*dt, final_time) for timestep in range(super_timestep)]) 
        kernel_integrand = kernel_vals.reshape(-1, 1) * permenant_price_impact_vals * dt
        # We do a vstack as the axis=1 are the batches
        kernel_integrand = np.vstack([inital_prices, kernel_integrand])
        kernel_integral = np.sum(kernel_integrand, axis=0)

        # This gives K2(s, t) for s<=t to integrate over
        kernel_vals_vol = np.array([vol_kernel_function(timestep*dt, final_time) for timestep in range(super_timestep)]).reshape(-1, 1)
        kernel_integrand_vol = sigma*kernel_vals_vol*(dt**0.5)*noise_process[:super_timestep, :]
        kernel_integral_vol = np.sum(kernel_integrand_vol, axis=0)
        
        s_t = kernel_integral + kernel_integral_vol

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