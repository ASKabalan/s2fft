from jax import jit

import numpy as np
import jax.numpy as jnp
import torch
from s2fft.utils import resampling, resampling_jax, resampling_torch
from s2fft.utils import healpix_ffts as hp
from s2fft.sampling import so3_samples as samples
from functools import partial


def inverse(
    flmn: np.ndarray,
    L: int,
    N: int,
    kernel: np.ndarray = None,
    sampling: str = "mw",
    reality: bool = False,
    method: str = "jax",
    nside: int = None,
) -> np.ndarray:
    r"""Compute the inverse Wigner transform, i.e. inverse Fourier transform on
    :math:`SO(3)`.

    Args:
        flm (np.ndarray): Wigner coefficients with shape :math:`[2N-1, L, 2L-1]`.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        kernel (np.ndarray, optional): Wigner-d kernel. Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}. Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        method (str, optional): Execution mode in {"numpy", "jax", "torch"}.
            Defaults to "jax".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray:  Signal on the on :math:`SO(3)` with shape :math:`[n_{\gamma},
        n_{\beta}, n_{\alpha}]`, where :math:`n_\xi` denotes the number of samples for
        angle :math:`\xi`.

    Note:
        Importantly, the convention adopted for storage of f is :math:`[\gamma, \beta,
        \alpha]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
        :math:`zyz` Euler convention, in order to simplify indexing for internal use.
        For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
        :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
        :math:`\alpha` with :math:`\phi`.
    """
    if method == "numpy":
        return inverse_transform(flmn, kernel, L, N, sampling, reality, nside)
    elif method == "jax":
        return inverse_transform_jax(flmn, kernel, L, N, sampling, reality, nside)
    elif method == "torch":
        return inverse_transform_torch(flmn, kernel, L, N, sampling, reality, nside)
    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(
    flmn: np.ndarray,
    kernel: np.ndarray,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    nside: int,
) -> np.ndarray:
    r"""Compute the inverse Wigner transform, i.e. inverse Fourier transform on
    :math:`SO(3)`.

    Args:
        flmn (np.ndarray): Wigner coefficients with shape :math:`[2N-1, L, 2L-1]`.

        kernel (np.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Pixel-space coefficients.
    """
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0
    n_start_ind = N - 1 if reality else 0

    fnab = np.zeros(samples.fnab_shape(L, N, sampling, nside), dtype=np.complex128)
    fnab[n_start_ind:, :, m_offset:] = np.einsum(
        "...ntlm, ...nlm -> ...ntm", kernel, flmn[n_start_ind:, :, :]
    )

    if sampling.lower() in "healpix":
        f = np.zeros(samples.f_shape(L, N, sampling, nside), dtype=np.complex128)
        for n in range(n_start_ind - N + 1, N):
            ind = N - 1 + n
            f[ind] = hp.healpix_ifft(fnab[ind], L, nside, "numpy")
        if reality:
            return np.fft.irfft(f[n_start_ind:], 2 * N - 1, axis=-2, norm="forward")
        else:
            return np.fft.ifft(np.fft.ifftshift(f, axes=-2), axis=-2, norm="forward")

    else:
        if reality:
            fnab = np.fft.ifft(np.fft.ifftshift(fnab, axes=-1), axis=-1, norm="forward")
            return np.fft.irfft(fnab[n_start_ind:], 2 * N - 1, axis=-3, norm="forward")
        else:
            fnab = np.fft.ifftshift(fnab, axes=(-1, -3))
            return np.fft.ifft2(fnab, axes=(-1, -3), norm="forward")


@partial(jit, static_argnums=(2, 3, 4, 5, 6))
def inverse_transform_jax(
    flmn: jnp.ndarray,
    kernel: jnp.ndarray,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    nside: int,
) -> jnp.ndarray:
    r"""Compute the inverse Wigner transform, i.e. inverse Fourier transform on
    :math:`SO(3)`.

    Args:
        flmn (jnp.ndarray): Wigner coefficients with shape :math:`[2N-1, L, 2L-1]`.

        kernel (jnp.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        jnp.ndarray: Pixel-space coefficients.
    """
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0
    n_start_ind = N - 1 if reality else 0

    fnab = jnp.zeros(samples.fnab_shape(L, N, sampling, nside), dtype=jnp.complex128)
    fnab = fnab.at[n_start_ind:, :, m_offset:].set(
        jnp.einsum(
            "...ntlm, ...nlm -> ...ntm",
            kernel,
            flmn[n_start_ind:, :, :],
            optimize=True,
        )
    )

    if sampling.lower() in "healpix":
        f = jnp.zeros(samples.f_shape(L, N, sampling, nside), dtype=jnp.complex128)
        for n in range(n_start_ind - N + 1, N):
            ind = N - 1 + n
            f = f.at[ind].set(hp.healpix_ifft(fnab[ind], L, nside, "jax"))
        if reality:
            return jnp.fft.irfft(f[n_start_ind:], 2 * N - 1, axis=-2, norm="forward")
        else:
            return jnp.conj(
                jnp.fft.fft(
                    jnp.fft.ifftshift(jnp.conj(f), axes=-2),
                    axis=-2,
                    norm="backward",
                )
            )

    else:
        if reality:
            fnab = jnp.conj(
                jnp.fft.fft(
                    jnp.fft.ifftshift(jnp.conj(fnab), axes=-1),
                    axis=-1,
                    norm="backward",
                )
            )
            return jnp.fft.irfft(fnab[n_start_ind:], 2 * N - 1, axis=-3, norm="forward")
        else:
            fnab = jnp.conj(jnp.fft.ifftshift(fnab, axes=(-1, -3)))
            return jnp.conj(jnp.fft.fft2(fnab, axes=(-1, -3), norm="backward"))


def inverse_transform_torch(
    flmn: torch.tensor,
    kernel: torch.tensor,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    nside: int,
) -> torch.tensor:
    r"""Compute the inverse Wigner transform, i.e. inverse Fourier transform on
    :math:`SO(3)`.

    Args:
        flmn (torch.tensor): Wigner coefficients with shape :math:`[2N-1, L, 2L-1]`.

        kernel (torch.tensor): Wigner-d kernel.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        torch.tensor: Pixel-space coefficients.
    """
    m_offset = 1 if sampling in ["mwss", "healpix"] else 0
    n_start_ind = N - 1 if reality else 0

    fnab = torch.zeros(
        samples.fnab_shape(L, N, sampling, nside), dtype=torch.complex128
    )
    if sampling.lower() == "healpix":
        fnab[n_start_ind:, :, m_offset:] = torch.einsum(
            "...ntlm, ...nlm -> ...ntm", kernel, flmn[n_start_ind:, :, :]
        )
    else:
        fnab[n_start_ind:, :, m_offset:].real = torch.einsum(
            "...ntlm, ...nlm -> ...ntm", kernel, flmn[n_start_ind:, :, :].real
        )
        fnab[n_start_ind:, :, m_offset:].imag = torch.einsum(
            "...ntlm, ...nlm -> ...ntm", kernel, flmn[n_start_ind:, :, :].imag
        )

    if sampling.lower() in "healpix":
        f = torch.zeros(samples.f_shape(L, N, sampling, nside), dtype=torch.complex128)
        for n in range(n_start_ind - N + 1, N):
            ind = N - 1 + n
            f[ind] = hp.healpix_ifft(fnab[ind], L, nside, "torch")
        if reality:
            return torch.fft.irfft(f[n_start_ind:], 2 * N - 1, axis=-2, norm="forward")
        else:
            return torch.fft.ifft(
                torch.fft.ifftshift(f, dim=[-2]), axis=-2, norm="forward"
            )

    else:
        if reality:
            fnab = torch.fft.ifft(
                torch.fft.ifftshift(fnab, dim=[-1]), axis=-1, norm="forward"
            )
            return torch.fft.irfft(
                fnab[n_start_ind:], 2 * N - 1, axis=-3, norm="forward"
            )
        else:
            fnab = torch.fft.ifftshift(fnab, dim=[-1, -3])
            return torch.fft.ifft2(fnab, dim=[-1, -3], norm="forward")


def forward(
    f: np.ndarray,
    L: int,
    N: int,
    kernel: np.ndarray = None,
    sampling: str = "mw",
    reality: bool = False,
    method: str = "jax",
    nside: int = None,
) -> np.ndarray:
    r"""Compute the forward Wigner transform, i.e. Fourier transform on
    :math:`SO(3)`.

    Args:
        f (np.ndarray): Signal on the on :math:`SO(3)` with shape
            :math:`[n_{\gamma}, n_{\beta}, n_{\alpha}]`, where :math:`n_\xi` denotes the
            number of samples for angle :math:`\xi`.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        kernel (np.ndarray, optional): Wigner-d kernel. Defaults to None.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}. Defaults to "mw".

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.
            Defaults to False.

        method (str, optional): Execution mode in {"numpy", "jax", "torch"}.
            Defaults to "jax".

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Wigner coefficients `flmn` with shape :math:`[2N-1, L, 2L-1]`.

    Note:
        Importantly, the convention adopted for storage of f is :math:`[\gamma, \beta,
        \alpha]`, for Euler angles :math:`(\alpha, \beta, \gamma)` following the
        :math:`zyz` Euler convention, in order to simplify indexing for internal use.
        For a given :math:`\gamma` we thus recover a signal on the sphere indexed by
        :math:`[\theta, \phi]`, i.e. we associate :math:`\beta` with :math:`\theta` and
        :math:`\alpha` with :math:`\phi`.
    """
    if method == "numpy":
        return forward_transform(f, kernel, L, N, sampling, reality, nside)
    elif method == "jax":
        return forward_transform_jax(f, kernel, L, N, sampling, reality, nside)
    elif method == "torch":
        return forward_transform_torch(f, kernel, L, N, sampling, reality, nside)
    else:
        raise ValueError(f"Method {method} not recognised.")


def forward_transform(
    f: np.ndarray,
    kernel: np.ndarray,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    nside: int,
) -> np.ndarray:
    r"""Compute the forward Wigner transform, i.e. Fourier transform on
    :math:`SO(3)`.

    Args:
        f (np.ndarray): Signal on the sphere.

        kernel (np.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        np.ndarray: Wigner space coefficients.
    """
    n_start_ind = N - 1 if reality else 0

    ax = -2 if sampling.lower() == "healpix" else -3
    if reality:
        fban = np.fft.rfft(np.real(f), axis=ax, norm="backward")
    else:
        fban = np.fft.fftshift(np.fft.fft(f, axis=ax, norm="backward"), axes=ax)

    spins = -np.arange(n_start_ind - N + 1, N)
    if sampling.lower() == "mw":
        fban = resampling.mw_to_mwss(fban, L, spins)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        fban = resampling.upsample_by_two_mwss(fban, L, spins)

    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    if sampling.lower() in "healpix":
        temp = np.zeros(samples.fnab_shape(L, N, sampling, nside), dtype=np.complex128)
        for n in range(n_start_ind - N + 1, N):
            ind = n if reality else N - 1 + n
            temp[N - 1 + n] = hp.healpix_fft(fban[ind], L, nside, "numpy")
        fban = temp[n_start_ind:, :, m_offset:]

    else:
        fban = np.fft.fft(fban, axis=-1, norm="backward")
        fban = np.fft.fftshift(fban, axes=-1)[:, :, m_offset:]

    flmn = np.zeros(samples.flmn_shape(L, N), dtype=np.complex128)
    flmn[n_start_ind:] = np.einsum("...ntlm, ...ntm -> ...nlm", kernel, fban)
    if reality:
        flmn[:n_start_ind] = np.conj(np.flip(flmn[n_start_ind + 1 :], axis=(-1, -3)))
        flmn[:n_start_ind] = np.einsum(
            "...nlm,...m->...nlm",
            flmn[:n_start_ind],
            (-1) ** abs(np.arange(-L + 1, L)),
        )
        flmn[:n_start_ind] = np.einsum(
            "...nlm,...n->...nlm",
            flmn[:n_start_ind],
            (-1) ** abs(np.arange(-N + 1, 0)),
        )

    return flmn


@partial(jit, static_argnums=(2, 3, 4, 5, 6))
def forward_transform_jax(
    f: jnp.ndarray,
    kernel: jnp.ndarray,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    nside: int,
) -> jnp.ndarray:
    r"""Compute the forward Wigner transform, i.e. Fourier transform on
    :math:`SO(3)`.

    Args:
        f (jnp.ndarray): Signal on the sphere.

        kernel (jnp.ndarray): Wigner-d kernel.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        jnp.ndarray: Wigner space coefficients.
    """
    n_start_ind = N - 1 if reality else 0

    ax = -2 if sampling.lower() == "healpix" else -3
    if reality:
        fban = jnp.fft.rfft(jnp.real(f), axis=ax, norm="backward")
    else:
        fban = jnp.fft.fftshift(jnp.fft.fft(f, axis=ax, norm="backward"), axes=ax)

    spins = -jnp.arange(n_start_ind - N + 1, N)
    if sampling.lower() == "mw":
        fban = resampling_jax.mw_to_mwss(fban, L, spins)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        fban = resampling_jax.upsample_by_two_mwss(fban, L, spins)

    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    if sampling.lower() in "healpix":
        temp = jnp.zeros(
            samples.fnab_shape(L, N, sampling, nside), dtype=jnp.complex128
        )
        for n in range(n_start_ind - N + 1, N):
            ind = n if reality else N - 1 + n
            temp = temp.at[N - 1 + n].set(hp.healpix_fft(fban[ind], L, nside, "jax"))
        fban = temp[n_start_ind:, :, m_offset:]

    else:
        fban = jnp.fft.fft(fban, axis=-1, norm="backward")
        fban = jnp.fft.fftshift(fban, axes=-1)[:, :, m_offset:]

    flmn = jnp.zeros(samples.flmn_shape(L, N), dtype=jnp.complex128)
    flmn = flmn.at[n_start_ind:].set(
        jnp.einsum("...ntlm, ...ntm -> ...nlm", kernel, fban, optimize=True)
    )
    if reality:
        flmn = flmn.at[:n_start_ind].set(
            jnp.conj(jnp.flip(flmn[n_start_ind + 1 :], axis=(-1, -3)))
        )
        flmn = flmn.at[:n_start_ind].set(
            jnp.einsum(
                "...nlm,...m->...nlm",
                flmn[:n_start_ind],
                (-1) ** abs(jnp.arange(-L + 1, L)),
                optimize=True,
            )
        )
        flmn = flmn.at[:n_start_ind].set(
            jnp.einsum(
                "...nlm,...n->...nlm",
                flmn[:n_start_ind],
                (-1) ** abs(jnp.arange(-N + 1, 0)),
                optimize=True,
            )
        )

    return flmn


def forward_transform_torch(
    f: torch.tensor,
    kernel: torch.tensor,
    L: int,
    N: int,
    sampling: str,
    reality: bool,
    nside: int,
) -> torch.tensor:
    r"""Compute the forward Wigner transform, i.e. Fourier transform on
    :math:`SO(3)`.

    Args:
        f (torch.tensor): Signal on the sphere.

        kernel (torch.tensor): Wigner-d kernel.

        L (int): Harmonic band-limit.

        N (int): Directional band-limit.

        sampling (str): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh", "healpix"}.

        reality (bool, optional): Whether the signal on the sphere is real.  If so,
            conjugate symmetry is exploited to reduce computational costs.

        nside (int): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".

    Returns:
        torch.tensor: Wigner space coefficients.
    """
    n_start_ind = N - 1 if reality else 0

    ax = -2 if sampling.lower() == "healpix" else -3
    if reality:
        fban = torch.fft.rfft(torch.real(f), axis=ax, norm="backward")
    else:
        fban = torch.fft.fftshift(torch.fft.fft(f, axis=ax, norm="backward"), dim=ax)

    spins = -torch.arange(n_start_ind - N + 1, N)
    if sampling.lower() == "mw":
        fban = resampling_torch.mw_to_mwss(fban, L, spins)

    if sampling.lower() in ["mw", "mwss"]:
        sampling = "mwss"
        fban = resampling_torch.upsample_by_two_mwss(fban, L, spins)

    m_offset = 1 if sampling in ["mwss", "healpix"] else 0

    if sampling.lower() in "healpix":
        temp = torch.zeros(
            samples.fnab_shape(L, N, sampling, nside), dtype=torch.complex128
        )
        for n in range(n_start_ind - N + 1, N):
            ind = n if reality else N - 1 + n
            temp[N - 1 + n] = hp.healpix_fft(fban[ind], L, nside, "torch")
        fban = temp[n_start_ind:, :, m_offset:]

    else:
        fban = torch.fft.fft(fban, axis=-1, norm="backward")
        fban = torch.fft.fftshift(fban, dim=[-1])[:, :, m_offset:]

    flmn = torch.zeros(samples.flmn_shape(L, N), dtype=torch.complex128)

    if sampling.lower() == "healpix":
        flmn[n_start_ind:] = torch.einsum("...ntlm, ...ntm -> ...nlm", kernel, fban)
    else:
        flmn[n_start_ind:].real = torch.einsum(
            "...ntlm, ...ntm -> ...nlm", kernel, fban.real
        )
        flmn[n_start_ind:].imag = torch.einsum(
            "...ntlm, ...ntm -> ...nlm", kernel, fban.imag
        )
    if reality:
        flmn[:n_start_ind] = torch.conj(
            torch.flip(flmn[n_start_ind + 1 :], dims=(-1, -3))
        )
        flmn[:n_start_ind] = torch.einsum(
            "...nlm,...m->...nlm",
            flmn[:n_start_ind],
            (-1) ** abs(torch.arange(-L + 1, L)),
        )
        flmn[:n_start_ind] = torch.einsum(
            "...nlm,...n->...nlm",
            flmn[:n_start_ind],
            (-1) ** abs(torch.arange(-N + 1, 0)),
        )

    return flmn
