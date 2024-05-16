import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

from s2fft.utils.healpix_ffts import spectral_folding_jax, spectral_periodic_extension_jax, healpix_fft_jax, healpix_ifft_jax

nside = 4
L = 2 * nside
total_pixels = 12 * nside**2
ftm_shape = (4 * nside - 1, 2 * L)
ftm_size = ftm_shape[0] * ftm_shape[1]

#arr = jnp.arange(total_pixels)
arr = jnp.arange(ftm_size)
arr = arr.reshape(ftm_shape)
res = healpix_ifft_jax(arr, L, nside, reality=False)

for i, elem in enumerate(res.flatten()):
    print(f"[{i}] {elem.real} + {elem.imag}j")
