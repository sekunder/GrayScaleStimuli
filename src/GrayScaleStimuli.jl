"""
Stimulus

A wrapper class and API for handling stimuli without needing to worry about the
specific implementation for that experiment. Basically, I want to be able to
call `compute_STRFs(raster, stimulus)` regardless of the data source.
"""
module GrayScaleStimuli

using JLD
import Base: show, isequal, hash
include("metadata.jl")

################################################################################
#### Abstract type to use as a placeholder/to get all docstrings
################################################################################
""""
    AbstractStimulus

Abstract type used as placeholder.
"""
abstract type AbstractStimulus end

################################################################################
#### Miscellaneous functions and constants
################################################################################

_NI(m) = error("Not Implemented: $m")

"""
    frame_size(S)

The size of the image, in pixels. Returns a tuple `(w,h)`; each frame of the image is
displayed on screen as an image that many pixels in size.
"""
frame_size(S::AbstractStimulus) = _NI("frame_size($(typeof(S)))")


"""
    frame_time(S)

The duration of a single frame, in seconds
"""
frame_time(S::AbstractStimulus) = _NI("frame_time($(typeof(S)))")

"""
    frame_rate(S::AbstractStimulus)

Convenience method returns `1/frame_time(S)`. Each concrete subtype of
`AbstractStimulus` should implement `frame_time`.
"""
frame_rate(S::AbstractStimulus) = 1.0 / frame_time(S)

"""
    n_frames(S)

The number of frames
"""
n_frames(S::AbstractStimulus) = _NI("n_frames($(typeof(S)))")

"""
    resolution(S::AbstractStimulus)

The resolution of the individual frames
"""
resolution(S::AbstractStimulus) = _NI("resolution($(typeof(S)))")

"""
    patch_size(S::AbstractStimulus)

The size, in pixels, of the individual patches of constant value. For any stimulus `S`, we
should have `frame_size(S) == resolution(S) .* patch_size(S)`
"""
patch_size(S::AbstractStimulus) = _NI("patch_size($(typeof(S)))")

"""
    matrix_form([Float64], S::AbstractStimulus, frames=1:n_frames(S))

A minimal representation of `S` as a `Matrix` of the specified type. The `i,j`th entry is the `i`th pixel of the `j`th frame.
"""
matrix_form(::Type{T}, S::AbstractStimulus, frames=1:n_frames(S)) where T = _NI("matrix_form($T, $(tpyoef(S)), $frames)")
matrix_form(S::AbstractStimulus, frames=1:n_frames(S)) = matrix_form(Float64, S, frames)


"""
    _compute_STRFs(spike_hist, stim; kwargs...)

Generic implementation of STRF computation. Uses `matrix_form(stim)` to perform
computations. Returns a matrix.

Life will be easy if we assume that the bins of the histogram are 1-to-1 with
the frames of the stimulus. So, implement concrete subtypes of
`AbstractStimulus` with that in mind.

For keyword arguments: specifying `window_length_frames` overrides any value
computed from `window_length_s`
"""
function _compute_STRFs(spike_hist::Matrix{Float64}, stim::AbstractStimulus; kwargs...)
    # preprocessing stuff: get the window length
    dkwargs = Dict(kwargs)
    window_length_s = get(dkwargs, :window_length_s, 0.5)
    window_length_frames = round(Int, window_length_s / frame_time(stim))
    if haskey(dkwargs, :window_length_frames)
        window_length_frames = dkwargs[:window_length_frames]
    end

    # get the matrix form of the stimulus
    stimulus = matrix_form(stim)

    N_bins, N_cells = size(spike_hist)
    N_bits_per_frame, N_frames = size(stimulus)
    if N_bins != N_frames
        error("_compute_STRFs: mismatched histogram and matrix form (N_bins = $N_bins, N_frames = $N_frames)")
    end

    avg_spikes_per_frame = zeros(spike_hist)
    for j = 1:N_cells
        avg_spikes_per_frame[window_length_frames:end,j] = spike_hist[window_length_frames:end,j] / sum(spike_hist[window_length_frames:end-1,j])
    end

    RFs = zeros(N_bits_per_frame, N_cells, window_length_frames)
    for k = 1:window_length_frames
        RFs[:, :, window_length_frames - k + 1] = stimulus * avg_spikes_per_frame
        avg_spikes_per_frame = circshift(avg_spikes_per_frame, [-1,0])
    end
    return RFs
end

################################################################################
#### Include definitions of concrete types
################################################################################
include("GrayScaleStimulus.jl")
include("optimizers.jl")
include("STRF_response.jl")
include("animation.jl")
include("fileio.jl")
# include("CRCNS/CRCNS_Stimulus.jl")
# include("animation.jl")
# include("Ganmor/Ganmor_stimulus.jl")

export show, isequal, hash, savestimulus, loadstimulus,
       frame_size, frame_time, frame_rate, n_frames, matrix_form, frame_image,
       time_to_index, index_to_time,
       GrayScaleStimulus, #CRCNS_Stimulus, Ganmor_Stimulus,
       compute_STRFs, STRF_response, scale_response,
       animated_gif

end
