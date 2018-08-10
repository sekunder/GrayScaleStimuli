
function isequal(S::AbstractStimulus, Q::AbstractStimulus)
    typeof(S) == typeof(Q) || return false
    for (fp,fq) in zip(fieldnames(typeof(S)),fieldnames(typeof(Q)))
        isequal(getfield(S, fp), getfield(Q, fq)) || return false
    end
    return true
end

function hash(S::AbstractStimulus, h::UInt)
    _h = h
    for fn in fieldnames(S)
        _h = hash(getfield(S, fn), _h)
    end
    return _h
end

################################################################################
#### File IO
################################################################################
"""
    savestimulus(S::AbstractStimulus; [filename], [dir])

Save the given stimulus to disk using the `JLD` package. Returns the full path to the
saved file.

Default `filename` is `string(hash(S))`. Default `dir` is `Pkg.dir(GrayScaleStimuli)/saved`.

If `dir` doesn't exist, will use `mkpath`. If file exists, contents will be overwritten.
"""
function savestimulus(S::AbstractStimulus;
    filename=string(hash(S)),
    dir=joinpath(Pkg.dir("GrayScaleStimuli"), "saved"))

    # _dir = abspath(dir)
    if !ispath(dir)
        mkpath(dir)
    end
    _fn = endswith(filename, ".jld") ? filename : filename * ".jld"
    # save(joinpath(_dir, _fn), "S", S)
    jldopen(joinpath(dir, _fn), "w") do file
        addrequire(file, GrayScaleStimuli)
        write(file, "S", S)
    end
    return joinpath(dir, _fn)
end

"""
    loadstimulus(filename; [dir])
    loadstimulus(h; [dir])

Returns the stimulus saved in `filename` (or identified by hash `h`) as saved by
`savestimulus`. Default `dir` is `Pkg.dir(GrayScaleStimuli)/saved`

"""
function loadstimulus(filename; dir=joinpath(Pkg.dir("GrayScaleStimuli"), "saved"))
    _fn = endswith(filename, ".jld") ? filename : filename * ".jld"
    _fn = basename(_fn)
    return load(joinpath(dir, _fn), "S")
end
loadstimulus(h::UInt; dir=joinpath(Pkg.dir("GrayScaleStimuli"),"saved")) = loadstimulus(string(h); dir=dir)
